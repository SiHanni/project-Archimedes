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
