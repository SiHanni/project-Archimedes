"""
골든 세트 회귀 + RMSE 집계.

`ARCHIMEDES_GOLDEN_ROOT` 가 가리키는 디렉터리에 manifest.json + 이미지가 있을 때만
실행된다(기본 pytest 에서는 `-m golden_manifest` 로 제외).

두 가지를 본다.
1. **케이스별 회귀**: 코드·모델을 바꿔도 각 케이스 결과가 허용 구간 안인가.
2. **집계 RMSE**: 실측(`actual_mass_g`)이 적힌 케이스들의 중량 RMSE 가
   매니페스트가 정한 상한 이하인가 (계획서 평가표 4번).

실사진은 PII·저작권 때문에 레포에 넣지 않는다 — `golden/README.md` 참고.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.eval import error_stats
from app.models.schemas import JobInputRecord, JobViews
from app.pipeline.exceptions import PipelineError
from app.pipeline.runner import SINGLE_VIEW_KEY, run_pipeline


def _golden_root() -> Path | None:
    raw = os.environ.get("ARCHIMEDES_GOLDEN_ROOT", "").strip()
    if not raw:
        return None
    p = Path(raw).resolve()
    return p if p.is_dir() else None


def _load_case(root: Path, case: dict[str, Any]) -> tuple[JobInputRecord, dict[str, bytes]]:
    """매니페스트 케이스 → (입력 레코드, 이미지 바이트)."""
    inp = case.get("input", {})
    mode = case.get("capture_mode") or ("multiview" if case.get("views") else "single")
    cid = case.get("id", "?")
    common = {
        "metal": inp.get("metal", "gold"),
        "purity": inp.get("purity", "18k"),
        "product_k": inp.get("product_k", "ring"),
    }

    images: dict[str, bytes] = {}
    if mode == "multiview":
        for vk, rel in dict(case["views"]).items():
            fp = root / rel
            assert fp.is_file(), f"case {cid}: missing {fp}"
            images[vk] = fp.read_bytes()
        record = JobInputRecord(
            capture_mode="multiview",
            views=JobViews(front="g", top="g", left="g", right="g", back="g"),
            **common,
        )
    else:
        rel = case.get("image") or (case.get("views") or {}).get(SINGLE_VIEW_KEY)
        assert rel, f"case {cid}: single-mode case needs `image`"
        fp = root / rel
        assert fp.is_file(), f"case {cid}: missing {fp}"
        images[SINGLE_VIEW_KEY] = fp.read_bytes()
        record = JobInputRecord(capture_mode="single", image="g", **common)
    return record, images


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
    settings = Settings()

    data = json.loads(mf.read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    assert cases, "manifest has no cases"

    predicted: list[float] = []
    actual: list[float] = []
    depth_rmse: list[float] = []
    failures: list[str] = []

    for case in cases:
        cid = case.get("id", "?")
        record, images = _load_case(root, case)

        expect_err = case.get("expect_error")
        if expect_err:
            with pytest.raises(PipelineError) as ei:
                run_pipeline(f"golden-{cid}", record, images, settings)
            assert ei.value.code == expect_err, f"{cid}: got {ei.value.code}"
            continue

        out = run_pipeline(f"golden-{cid}", record, images, settings)
        m = float(out["mass_est_g"])
        exp = case.get("expect", {})

        if "mass_est_g_min" in exp and not (exp["mass_est_g_min"] <= m <= exp["mass_est_g_max"]):
            failures.append(
                f"{cid}: mass {m:.4f} not in [{exp['mass_est_g_min']}, {exp['mass_est_g_max']}]"
            )
        if exp.get("tier_in") and out["confidence_tier"] not in exp["tier_in"]:
            failures.append(f"{cid}: tier {out['confidence_tier']} not in {exp['tier_in']}")

        if case.get("actual_mass_g"):
            predicted.append(m)
            actual.append(float(case["actual_mass_g"]))

        rmse = ((out.get("meta") or {}).get("scale_fusion") or {}).get("depth_rmse_mm")
        if isinstance(rmse, (int, float)):
            depth_rmse.append(float(rmse))

    # 케이스 단위 실패를 한 번에 보고한다 — 첫 실패에서 멈추면 전체 상태를 못 본다
    assert not failures, "golden case failures:\n  " + "\n  ".join(failures)

    if predicted:
        stats = error_stats(predicted, actual)
        print(f"\n[golden] 중량 {stats.as_dict()}")
        cap = data.get("expect_rmse_g_max")
        if cap is not None:
            assert stats.rmse <= float(cap), (
                f"mass RMSE {stats.rmse:.4f} g exceeds manifest cap {cap} g"
            )

    if depth_rmse:
        print(f"[golden] 거리 홀드아웃 RMSE mm: n={len(depth_rmse)} max={max(depth_rmse):.3f}")
        cap = data.get("expect_depth_rmse_mm_max")
        if cap is not None:
            assert max(depth_rmse) <= float(cap), (
                f"depth RMSE {max(depth_rmse):.3f} mm exceeds manifest cap {cap} mm"
            )
