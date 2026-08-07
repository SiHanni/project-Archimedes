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
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

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


# BRPOP 대기 시간. 소켓 타임아웃은 이보다 넉넉해야 한다.
_BRPOP_TIMEOUT_S = 5


def main() -> None:
    settings = get_settings()
    # socket_timeout 을 BRPOP 대기보다 길게 잡지 않으면, 큐가 비어 있는 **정상 상황**마다
    # 소켓 읽기 타임아웃 예외가 올라와 로그가 스택트레이스로 뒤덮인다.
    # (동작에는 문제가 없지만 진짜 에러를 가려 버린다)
    r = redis.from_url(
        settings.redis_url,
        socket_timeout=_BRPOP_TIMEOUT_S + 10,
        socket_keepalive=True,
        health_check_interval=30,
    )
    q = settings.queue_name
    log.info("listening on %s", q)
    while True:
        try:
            item = r.brpop(q, timeout=_BRPOP_TIMEOUT_S)
            if not item:
                continue
            _, job_id_bytes = item
            job_id = job_id_bytes.decode("utf-8")
            process_one(job_id)
        except KeyboardInterrupt:
            sys.exit(0)
        except RedisTimeoutError:
            # 큐가 비어 대기 시간이 만료된 것 — 정상 흐름이다
            continue
        except RedisError as e:
            # 브로커 재기동 등 — 스택트레이스 없이 한 줄로
            log.warning("redis error, retrying: %s", e)
            time.sleep(1.0)
        except Exception:
            log.exception("loop error")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
