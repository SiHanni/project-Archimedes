"""Synthetic desk scene — card + jewel beside card (§5.2)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.models.schemas import JobInputRecord, JobViews
from app.pipeline.card import compute_card_geometry
from app.pipeline.exceptions import PipelineError
from app.pipeline.geometry_g1 import jewel_bbox_uv_mm
from app.pipeline.ingest import bytes_to_bgr
from app.pipeline.quality_gate import check_image_quality
from app.pipeline.runner import run_pipeline
from app.pipeline.segment import build_jewel_mask


def make_synthetic_jpeg() -> bytes:
    img = np.full((2048, 2048, 3), 225, np.uint8)
    card = np.array([[420, 380], [1380, 400], [1390, 980], [400, 960]], np.int32)
    cv2.fillPoly(img, [card], (252, 252, 252))
    cv2.polylines(img, [card], True, (60, 60, 60), 3)
    cv2.circle(img, (1620, 680), 110, (12, 12, 18), -1)
    cv2.GaussianBlur(img, (3, 3), 0, dst=img)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buf.tobytes()


@pytest.mark.slow
def test_synthetic_front_view_chain(test_settings: Settings) -> None:
    """Card → jewel mask → mm bbox (one view)."""
    settings = test_settings
    raw = make_synthetic_jpeg()
    bgr = bytes_to_bgr(raw)
    check_image_quality(bgr, settings, "front")
    card = compute_card_geometry(bgr, "front")
    mask, _ = build_jewel_mask(bgr, card, settings, "front", job_id="t1")
    u0, u1, v0, v1 = jewel_bbox_uv_mm(mask, card, "front")
    assert u1 > u0 and v1 > v0, (u0, u1, v0, v1)


def test_same_jpeg_five_views_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    동일 파일을 5뷰에 넣는 것은 실사용에서 금지에 가깝다.
    v0 슬랩/카빙에 따라 수치가 나오거나 ERR_* 로 끝날 수 있음 — 크래시만 없으면 됨.
    """
    monkeypatch.setenv("ARCHIMEDES_MIN_SHORT_EDGE", "800")
    monkeypatch.setenv("ARCHIMEDES_BLUR_THRESHOLD", "3")
    monkeypatch.setenv("ARCHIMEDES_SCALE_MISMATCH", "0.25")
    monkeypatch.setenv("ARCHIMEDES_USE_VOXEL_CARVE", "0")
    settings = Settings()
    raw = make_synthetic_jpeg()
    images = {v: raw for v in ("front", "top", "left", "right", "back")}
    inp = JobInputRecord(
        views=JobViews(front="x", top="x", left="x", right="x", back="x"),
        metal="gold",
        purity="18k",
        product_k="ring",
    )
    try:
        out = run_pipeline("doc", inp, images, settings)
        assert "mass_est_g" in out
        assert "volume_model" in out.get("meta", {})
    except PipelineError as e:
        assert e.code in (
            "ERR_VOLUME",
            "ERR_SCALE_MISMATCH",
            "ERR_SILHOUETTE_AREA",
            "ERR_CARD_NOT_FOUND",
            "ERR_JEWEL_ON_CARD",
        )
