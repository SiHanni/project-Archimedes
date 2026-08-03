"""
Redis BRPOP worker — Phase 5 queue consumer.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback

import redis

from app.config import get_settings
from app.db import jobs as jobs_db
from app.models.schemas import JobInputRecord
from app.pipeline.exceptions import PipelineError
from app.pipeline.ingest import load_images_from_s3
from app.pipeline.runner import run_pipeline
from app.telemetry import maybe_init_otel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("archimedes.consumer")


def process_one(job_id: str) -> None:
    settings = get_settings()
    maybe_init_otel(settings)
    conn = jobs_db.connect(settings)
    try:
        row = jobs_db.fetch_job(conn, job_id)
        if not row:
            log.warning("unknown job %s", job_id)
            return
        if row["status"] not in ("pending", "processing"):
            log.info("skip job %s status=%s", job_id, row["status"])
            return

        inp_raw = row["input_json"]
        if isinstance(inp_raw, str):
            inp_raw = json.loads(inp_raw)
        inp = JobInputRecord.model_validate(inp_raw)

        jobs_db.mark_processing(conn, job_id, settings.algorithm_version)
        images = load_images_from_s3(settings, inp.image_keys())
        result = run_pipeline(job_id, inp, images, settings)
        jobs_db.mark_completed(conn, job_id, result)
        log.info("job %s completed", job_id)
    except PipelineError as pe:
        log.warning("job %s pipeline error %s", job_id, pe.code)
        payload = {
            "error": {
                "code": pe.code,
                "message": str(pe),
                "retry_step": pe.retry_step,
                "retry_views": pe.retry_views,
                "error_severity": pe.error_severity,
                "suggested_action": pe.suggested_action or "retry_one_view",
            },
            "meta": {
                "workflow": {
                    "error_severity": pe.error_severity,
                    "suggested_action": pe.suggested_action or "retry_one_view",
                    "retry_views": pe.retry_views,
                },
                "degraded_reasons": [f"{pe.code}:{v}" for v in pe.retry_views] or [pe.code],
            },
        }
        if pe.error_severity == "soft":
            jobs_db.mark_completed_low_confidence(conn, job_id, payload, pe.code, str(pe))
        else:
            jobs_db.mark_failed(conn, job_id, pe.code, str(pe), payload)
    except Exception:
        log.exception("job %s failed", job_id)
        jobs_db.mark_failed(conn, job_id, "ERR_INTERNAL", traceback.format_exc()[:2000])
    finally:
        conn.close()


def main() -> None:
    settings = get_settings()
    r = redis.from_url(settings.redis_url)
    q = settings.queue_name
    log.info("listening on %s", q)
    while True:
        try:
            item = r.brpop(q, timeout=5)
            if not item:
                continue
            _, job_id_bytes = item
            job_id = job_id_bytes.decode("utf-8")
            process_one(job_id)
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception:
            log.exception("loop error")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
