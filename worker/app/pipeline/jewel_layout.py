"""
얇은 귀걸이·펜던트: Visual Hull이 실물 부피를 크게 과대추정함(§6 Hollow, §7 체인 유사 이슈).

카드 대비 실루엣 가로 비(픽셀)가 작을수록 추가 부피 스케일을 곱해 v0에서 체감 오차를 줄인다.
데이터·실측으로 `constants`의 구간·계수를 재튜닝한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.constants import (
    EARRING_LAYOUT_VOL_MULT_LARGE,
    EARRING_LAYOUT_VOL_MULT_MED,
    EARRING_LAYOUT_VOL_MULT_SMALL,
    EARRING_LAYOUT_VOL_MULT_TINY,
    EARRING_LAYOUT_VOL_MULT_XL,
    JEWEL_LAYOUT_CHAIN_BRACELET_MAX_RATIO,
    JEWEL_LAYOUT_CHAIN_THIN_MULT_K,
    JEWEL_LAYOUT_CHAIN_THIN_MULT_MAX,
    JEWEL_LAYOUT_CHAIN_THIN_MULT_MIN,
    JEWEL_LAYOUT_CHAIN_THIN_SIDE_MAX,
    JEWEL_TO_CARD_WIDTH_RATIO_T1,
    JEWEL_TO_CARD_WIDTH_RATIO_T2,
    JEWEL_TO_CARD_WIDTH_RATIO_T3,
    JEWEL_TO_CARD_WIDTH_RATIO_T4,
    LAYOUT_CORRECT_PRODUCTS,
    LAYOUT_VOL_MULT_LARGE,
    LAYOUT_VOL_MULT_MED,
    LAYOUT_VOL_MULT_SMALL,
    LAYOUT_VOL_MULT_TINY,
    LAYOUT_VOL_MULT_XL,
    VIEW_ORDER,
)
from app.pipeline.card import CardGeometry, card_edge_lengths_px

_CHAIN_LIKE = frozenset({"necklace", "chain", "bracelet"})


def _card_width_px(card: CardGeometry) -> float:
    """카드 긴 변(ID-1 85.60mm 축) 픽셀 길이 — `card.card_edge_lengths_px` 와 단일 소스."""
    return max(card_edge_lengths_px(card.quad_px)[0], 1.0)


def jewel_to_card_size_ratios(
    masks: dict[str, Any],
    cards: dict[str, CardGeometry],
) -> tuple[float, float]:
    """
    (r_max, r_min_side) — 카드 대비 실루엣 크기.
    r_max: max(가로/카드너비) 뷰 중 최댓값 — 넓게 퍼진 마스크.
    r_min_side: min(min(가,세)/카드너비) 뷰 중 최솟값 — 귀걸이처럼 한 축이 아주 작을 때.
    """
    r_max_best = 0.0
    r_min_side_best = 1e9
    any_pix = False
    for v in VIEW_ORDER:
        mask = masks[v]
        card = cards[v]
        ys, xs = np.where(mask > 0)
        if len(xs) < 8:
            continue
        any_pix = True
        jw = float(xs.max() - xs.min())
        jh = float(ys.max() - ys.min())
        cw = _card_width_px(card)
        r_max_best = max(r_max_best, jw / cw)
        r_side = min(jw, jh) / cw
        r_min_side_best = min(r_min_side_best, r_side)
    if not any_pix:
        return 0.0, 0.0
    if r_min_side_best >= 1e8:
        r_min_side_best = 0.0
    return r_max_best, r_min_side_best


def _ratio_for_layout(product_k: str, r_max: float, r_min_side: float) -> float:
    """가느다란 물체: 짧은 변이 카드 대비 작으면 r_max만으로는 구간이 너무 커지는 것을 완화."""
    if product_k.lower() in _CHAIN_LIKE:
        return r_max
    if r_min_side <= 0:
        return r_max
    return float(min(r_max, r_min_side * 2.5))


def layout_volume_multiplier(product_k: str, r_max: float, r_min_side: float) -> tuple[float, dict[str, Any]]:
    """
    V_adj 최종에 곱할 계수(1.0 = 변경 없음).
    - 귀걸이·펜던트·반지·기타: 카드 대비 작으면 항상 적용(형태 오선택 대비).
    - 목걸이·체인·팔찌: (1) 실루엣 짧은 축이 카드 대비 작으면 thin 휴리스틱 적용
      (귀걸이를 체인으로 잘못 고른 경우 등). (2) r_max가 작으면 기존 버킷 테이블 적용.
    """
    pk = product_k.lower()
    ratio = _ratio_for_layout(pk, r_max, r_min_side)

    if pk in _CHAIN_LIKE:
        thin_side = r_min_side > 0 and r_min_side < JEWEL_LAYOUT_CHAIN_THIN_SIDE_MAX
        compact_spread = r_max > 0 and r_max <= JEWEL_LAYOUT_CHAIN_BRACELET_MAX_RATIO
        if not thin_side and not compact_spread:
            return 1.0, {
                "applied": False,
                "reason": "chain_like_skip_or_no_mask",
                "r_max": round(r_max, 5),
                "r_min_side": round(r_min_side, 5),
            }
        if thin_side:
            raw = JEWEL_LAYOUT_CHAIN_THIN_MULT_K * r_min_side
            mult = max(
                JEWEL_LAYOUT_CHAIN_THIN_MULT_MIN,
                min(JEWEL_LAYOUT_CHAIN_THIN_MULT_MAX, raw),
            )
            detail = {
                "applied": mult < 1.0,
                "r_max": round(r_max, 5),
                "r_min_side": round(r_min_side, 5),
                "ratio_effective": round(min(r_max, r_min_side * 2.5), 5),
                "bucket": "chain_thin_heuristic",
                "layout_volume_mult": mult,
            }
            return mult, detail
        ratio = r_max
        mult = 1.0
        bucket = "none"
        if ratio < JEWEL_TO_CARD_WIDTH_RATIO_T1:
            mult = LAYOUT_VOL_MULT_TINY
            bucket = "tiny"
        elif ratio < JEWEL_TO_CARD_WIDTH_RATIO_T2:
            mult = LAYOUT_VOL_MULT_SMALL
            bucket = "small"
        elif ratio < JEWEL_TO_CARD_WIDTH_RATIO_T3:
            mult = LAYOUT_VOL_MULT_MED
            bucket = "medium"
        elif ratio < JEWEL_TO_CARD_WIDTH_RATIO_T4:
            mult = LAYOUT_VOL_MULT_LARGE
            bucket = "large"
        else:
            mult = LAYOUT_VOL_MULT_XL
            bucket = "xl"
        detail = {
            "applied": mult < 1.0,
            "r_max": round(r_max, 5),
            "r_min_side": round(r_min_side, 5),
            "ratio_effective": round(ratio, 5),
            "bucket": bucket,
            "layout_volume_mult": mult,
        }
        return mult, detail
    elif pk not in LAYOUT_CORRECT_PRODUCTS:
        return 1.0, {"applied": False, "reason": "product_not_layout_corrected"}
    elif ratio <= 0.0:
        return 1.0, {"applied": False, "reason": "no_jewel_span"}

    mult = 1.0
    bucket = "none"
    earring_table = pk == "earring"
    if ratio < JEWEL_TO_CARD_WIDTH_RATIO_T1:
        mult = EARRING_LAYOUT_VOL_MULT_TINY if earring_table else LAYOUT_VOL_MULT_TINY
        bucket = "tiny"
    elif ratio < JEWEL_TO_CARD_WIDTH_RATIO_T2:
        mult = EARRING_LAYOUT_VOL_MULT_SMALL if earring_table else LAYOUT_VOL_MULT_SMALL
        bucket = "small"
    elif ratio < JEWEL_TO_CARD_WIDTH_RATIO_T3:
        mult = EARRING_LAYOUT_VOL_MULT_MED if earring_table else LAYOUT_VOL_MULT_MED
        bucket = "medium"
    elif ratio < JEWEL_TO_CARD_WIDTH_RATIO_T4:
        mult = EARRING_LAYOUT_VOL_MULT_LARGE if earring_table else LAYOUT_VOL_MULT_LARGE
        bucket = "large"
    else:
        mult = EARRING_LAYOUT_VOL_MULT_XL if earring_table else LAYOUT_VOL_MULT_XL
        bucket = "xl"

    detail: dict[str, Any] = {
        "applied": mult < 1.0,
        "r_max": round(r_max, 5),
        "r_min_side": round(r_min_side, 5),
        "ratio_effective": round(ratio, 5),
        "bucket": bucket,
        "layout_volume_mult": mult,
        "earring_layout_table": earring_table,
    }
    return mult, detail
