from __future__ import annotations

import boto3
from botocore.client import BaseClient

from app.config import Settings


def s3_client(settings: Settings) -> BaseClient:
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
    )


def download_object(settings: Settings, key: str) -> bytes:
    cli = s3_client(settings)
    resp = cli.get_object(Bucket=settings.s3_bucket, Key=key)
    return resp["Body"].read()


def upload_object(settings: Settings, key: str, data: bytes, content_type: str) -> str:
    """세그 산출물 등을 업로드하고 key 를 돌려준다."""
    cli = s3_client(settings)
    cli.put_object(Bucket=settings.s3_bucket, Key=key, Body=data, ContentType=content_type)
    return key
