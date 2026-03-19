"""
골든 세트 회귀: ARCHIMEDES_GOLDEN_ROOT 가 가리키는 디렉터리에 manifest.json + 이미지가 있을 때만 실행.
기본 pytest 에서는 제외됨 (-m golden_manifest).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.config import Settings
from app.models.schemas import JobInputRecord, JobViews
from app.pipeline.exceptions import PipelineError
from app.pipeline.runner import run_pipeline


def _golden_root() -> Path | None:
    raw = os.environ.get("ARCHIMEDES_GOLDEN_ROOT", "").strip()
    if not raw:
        return None
    p = Path(raw).resolve()
    return p if p.is_dir() else None


@pytest.mark.golden_manifest
def test_golden_manifest_runs_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _golden_root()
    if root is None:
        pytest.skip("Set ARCHIMEDES_GOLDEN_ROOT to golden/ directory")

    mf = root / "manifest.json"
    if not mf.is_file():
        pytest.skip(f"Missing {mf} (copy golden/manifest.example.json)")

    monkeypatch.setenv("ARCHIMEDES_MIN_SHORT_EDGE", "800")
    monkeypatch.setenv("ARCHIMEDES_BLUR_THRESHOLD", "3")
    monkeypatch.setenv("ARCHIMEDES_SCALE_MISMATCH", "0.25")
    monkeypatch.setenv("ARCHIMEDES_USE_VOXEL_CARVE", "0")
    settings = Settings()

    data = json.loads(mf.read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    assert cases, "manifest has no cases"

    for case in cases:
        cid = case.get("id", "?")
        views_paths: dict[str, str] = case["views"]
        inp = case["input"]
        exp = case["expect"]
        images: dict[str, bytes] = {}
        for vk, rel in views_paths.items():
            fp = root / rel
            assert fp.is_file(), f"case {cid}: missing {fp}"
            images[vk] = fp.read_bytes()

        rec = JobInputRecord(
            views=JobViews(
                front="g",
                top="g",
                left="g",
                right="g",
                back="g",
            ),
            metal=inp.get("metal", "gold"),
            purity=inp.get("purity", "18k"),
            product_k=inp.get("product_k", "ring"),
        )
        expect_err = case.get("expect_error")
        if expect_err:
            with pytest.raises(PipelineError) as ei:
                run_pipeline(f"golden-{cid}", rec, images, settings)
            assert ei.value.code == expect_err
            continue

        out = run_pipeline(f"golden-{cid}", rec, images, settings)
        m = out["mass_est_g"]
        assert exp["mass_est_g_min"] <= m <= exp["mass_est_g_max"], (
            f"{cid}: mass {m} not in [{exp['mass_est_g_min']}, {exp['mass_est_g_max']}]"
        )
        if exp.get("tier_in"):
            assert out["confidence_tier"] in exp["tier_in"], f"{cid}: tier {out['confidence_tier']}"
