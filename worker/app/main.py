from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings

app = FastAPI(title="Archimedes Worker", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/meta")
def meta() -> dict:
    s = get_settings()
    return {"algorithm_version": s.algorithm_version}
