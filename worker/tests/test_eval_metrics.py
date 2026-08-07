"""평가 지표 회귀 — 계획서 「평가방법 및 환경」 절차와 일치하는지."""

from __future__ import annotations

import math

import pytest

from app.eval import error_stats, group_error_stats


def test_rmse_follows_the_documented_procedure() -> None:
    """
    계획서 절차: (추정−실측) 제곱 → 합산 → 표본수로 나눔 → 루트.
    오차 [+1, -3, +2] → sqrt((1+9+4)/3) = sqrt(14/3).
    """
    st = error_stats([11.0, 7.0, 12.0], [10.0, 10.0, 10.0])
    assert st.n == 3
    assert st.rmse == pytest.approx(math.sqrt(14 / 3))
    assert st.mae == pytest.approx(2.0)


def test_perfect_prediction_is_zero() -> None:
    st = error_stats([5.0, 3.0], [5.0, 3.0])
    assert st.rmse == 0.0
    assert st.mape_pct == pytest.approx(0.0)


def test_bias_separates_systematic_over_estimation() -> None:
    """RMSE 만으로는 계통 편향을 못 본다 — α 가 통째로 크면 bias 가 양수로 뜬다."""
    over = error_stats([12.0, 11.0, 13.0], [10.0, 10.0, 10.0])
    mixed = error_stats([12.0, 8.0, 12.0], [10.0, 10.0, 10.0])
    assert over.bias > 1.5
    assert abs(mixed.bias) < 1.0


def test_mape_skips_zero_actuals_instead_of_dividing() -> None:
    st = error_stats([1.0, 2.0], [0.0, 1.0])
    assert st.mape_pct == pytest.approx(100.0)  # 두 번째 표본만 반영
    assert st.n == 2


def test_empty_input_is_safe() -> None:
    st = error_stats([], [])
    assert st.n == 0
    assert st.rmse == 0.0
    assert st.mape_pct is None


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        error_stats([1.0], [1.0, 2.0])


def test_group_stats_split_by_product_and_total() -> None:
    rows = [
        ("ring", 4.0, 5.0),
        ("ring", 6.0, 5.0),
        ("chain", 30.0, 20.0),
    ]
    got = group_error_stats(rows)
    assert set(got) == {"ring", "chain", "__all__"}
    assert got["ring"].n == 2
    assert got["chain"].rmse == pytest.approx(10.0)
    assert got["__all__"].n == 3


# ───────────────── 신뢰도 tier 규칙 ─────────────────


def test_precision_boost_cannot_claim_high_without_observed_thickness() -> None:
    """
    실측에서 발견: 폰 사진(EXIF 있음) + 기울인 카드면 precision_boost 가 켜져
    tier=high 가 됐는데, 같은 결과의 부피 불확실성은 σ≈0.5(두께 가정)였다.
    포즈 정밀도는 두께 미관측을 상쇄하지 못하므로 high 로 올리면 안 된다.
    """
    from app.pipeline.confidence import ConfidenceState

    assumed = ConfidenceState(
        multires_penalty=False,
        scale_tight=True,
        quality_ok=True,
        precision_boost=True,
        coarse_volume_model=True,  # 두께 클램프됨
    )
    assert assumed.tier() == "medium"

    observed = ConfidenceState(
        multires_penalty=False,
        scale_tight=True,
        quality_ok=True,
        precision_boost=True,
        coarse_volume_model=False,
    )
    assert observed.tier() == "high"


def test_boost_still_recovers_a_single_scale_deduction() -> None:
    """두께는 관측했는데 스케일만 헐거운 경우에는 boost 가 제 역할을 한다."""
    from app.pipeline.confidence import ConfidenceState

    s = ConfidenceState(
        multires_penalty=True,
        scale_tight=True,
        quality_ok=True,
        precision_boost=True,
        coarse_volume_model=False,
    )
    assert s.tier() == "high"
