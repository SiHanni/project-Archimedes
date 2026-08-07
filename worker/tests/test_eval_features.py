"""학습형 잔차 특징 추출 회귀 (§4.4)."""

from __future__ import annotations

import math

import pytest

from app.eval.features import (
    FEATURE_NAMES,
    apply_correction,
    extract_features,
    target_log_ratio,
)


def _result(**overrides) -> dict:
    base = {
        "mass_est_g": 4.0,
        "meta": {
            "capture_mode": "single",
            "reconstruction": {
                "method": "height_field",
                "area_proj_mm2": 300.0,
                "length_mm": 20.0,
                "width_mm": 10.0,
                "h_mean_mm": 1.5,
                "thickness_clamp": None,
            },
            "scale_fusion": {
                "anchor_used": True,
                "card_distance_mm": 200.0,
                "depth_rmse_mm": 2.0,
            },
        },
    }
    base["meta"].update(overrides.pop("meta", {}))
    base.update(overrides)
    return base


def test_feature_vector_matches_declared_names() -> None:
    """계수 파일과 순서가 어긋나면 보정이 통째로 틀어진다."""
    f = extract_features(_result())
    assert f is not None
    assert len(f) == len(FEATURE_NAMES)
    assert f[0] == 1.0  # bias


def test_multiview_results_are_excluded() -> None:
    """다뷰는 부피 산출 방식이 달라 같은 계수로 보정하면 안 된다."""
    r = _result()
    r["meta"]["capture_mode"] = "multiview"
    assert extract_features(r) is None


def test_degenerate_geometry_is_excluded() -> None:
    r = _result()
    r["meta"]["reconstruction"]["area_proj_mm2"] = 0.0
    assert extract_features(r) is None


def test_flags_are_encoded() -> None:
    clamped = _result()
    clamped["meta"]["reconstruction"]["thickness_clamp"] = "min"
    clamped["meta"]["scale_fusion"]["anchor_used"] = False

    idx_clamp = FEATURE_NAMES.index("thickness_assumed")
    idx_anchor = FEATURE_NAMES.index("anchor_used")
    f = extract_features(clamped)
    assert f is not None
    assert f[idx_clamp] == 1.0
    assert f[idx_anchor] == 0.0


def test_rel_depth_rmse_is_normalised_by_distance() -> None:
    idx = FEATURE_NAMES.index("rel_depth_rmse")
    f = extract_features(_result())
    assert f is not None
    assert f[idx] == pytest.approx(2.0 / 200.0)


def test_missing_fusion_does_not_crash() -> None:
    r = _result()
    r["meta"]["scale_fusion"] = {}
    f = extract_features(r)
    assert f is not None
    assert f[FEATURE_NAMES.index("rel_depth_rmse")] == 0.0


def test_target_is_log_ratio() -> None:
    assert target_log_ratio(8.0, 4.0) == pytest.approx(math.log(2.0))
    assert target_log_ratio(0.0, 4.0) is None
    assert target_log_ratio(4.0, 0.0) is None


def test_zero_weights_leave_mass_unchanged() -> None:
    """물리식이 베이스 — 학습이 없으면 원값 그대로여야 한다."""
    f = extract_features(_result())
    assert f is not None
    assert apply_correction(4.0, f, [0.0] * len(f)) == pytest.approx(4.0)


def test_length_mismatch_falls_back_to_original() -> None:
    """계수 파일이 낡아 길이가 안 맞으면 조용히 틀린 보정을 하지 말고 원값."""
    assert apply_correction(4.0, [1.0, 2.0], [0.5]) == pytest.approx(4.0)


def test_correction_is_clamped() -> None:
    """데이터 문제로 계수가 폭주해도 무게가 수십 배로 튀지 않는다."""
    f = [1.0] + [0.0] * (len(FEATURE_NAMES) - 1)
    w = [99.0] + [0.0] * (len(FEATURE_NAMES) - 1)
    got = apply_correction(4.0, f, w)
    assert got == pytest.approx(4.0 * math.exp(1.4))


def test_correction_is_always_positive() -> None:
    """로그 공간이라 어떤 계수에서도 음수 무게가 나오지 않는다."""
    f = extract_features(_result())
    assert f is not None
    w = [-50.0] * len(f)
    assert apply_correction(4.0, f, w) > 0
