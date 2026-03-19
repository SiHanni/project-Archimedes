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
