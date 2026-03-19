from __future__ import annotations

import json
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from app.config import Settings


def connect(settings: Settings):
    return pymysql.connect(cursorclass=DictCursor, **settings.mysql_dsn_kwargs)


def fetch_job(conn, job_id: str) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, status, input_json, algorithm_version FROM jobs WHERE id = %s",
            (job_id,),
        )
        row = cur.fetchone()
    return row


def mark_processing(conn, job_id: str, algo: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='processing', algorithm_version=%s WHERE id=%s",
            (algo, job_id),
        )
    conn.commit()


def mark_completed(conn, job_id: str, result: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='completed', result_json=%s, error_code=NULL, error_message=NULL WHERE id=%s",
            (json.dumps(result), job_id),
        )
    conn.commit()


def mark_failed(conn, job_id: str, code: str, message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='failed', error_code=%s, error_message=%s WHERE id=%s",
            (code, message, job_id),
        )
    conn.commit()
