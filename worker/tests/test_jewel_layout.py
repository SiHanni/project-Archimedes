"""jewel_layout: 체인류 thin 휴리스틱·버킷 분기."""

from __future__ import annotations

from app.pipeline.jewel_layout import layout_volume_multiplier


def test_chain_thin_side_gets_heuristic_mult() -> None:
    """r_max는 크지만 r_min_side가 작으면(가느다란 물체) 부피 스케일 적용."""
    mult, d = layout_volume_multiplier("chain", r_max=0.55, r_min_side=0.08)
    assert d["bucket"] == "chain_thin_heuristic"
    assert mult < 1.0
    assert 0.1 <= mult <= 0.6


def test_chain_wide_spread_skips() -> None:
    """길게 펼친 체인: 짧은 축도 크고 r_max도 크면 보정 생략."""
    mult, d = layout_volume_multiplier("chain", r_max=0.6, r_min_side=0.2)
    assert mult == 1.0
    assert d.get("reason") == "chain_like_skip_or_no_mask"


def test_earring_uses_tighter_layout_table() -> None:
    """귀걸이는 동일 ratio에서도 layout_mult가 일반 펜던트·반지보다 작다."""
    m_ear, d_ear = layout_volume_multiplier("earring", r_max=0.11, r_min_side=0.04)
    m_ring, _ = layout_volume_multiplier("ring", r_max=0.11, r_min_side=0.04)
    assert d_ear.get("earring_layout_table") is True
    assert m_ear < m_ring


def test_necklace_compact_uses_bucket() -> None:
    # 짧은 축이 THIN 임계 이상이면 thin 분기 대신 r_max 버킷
    mult, d = layout_volume_multiplier("necklace", r_max=0.12, r_min_side=0.16)
    assert d["bucket"] in ("small", "medium", "tiny")
    assert mult < 1.0
