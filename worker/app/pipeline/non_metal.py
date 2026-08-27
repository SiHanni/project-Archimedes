"""
누끼에서 **귀금속이 아닌 것**을 떨어낸다 (계획서 Step 1 후처리).

## 왜 필요한가

BiRefNet 은 "화면에서 두드러진 물체"를 잡지, "귀금속"을 잡지 않는다. 그래서
아크릴 봉인 케이스·주얼리 케이스·전자저울·보증서가 제품과 함께 딸려 온다.
크롭 재추론으로 용기를 배경으로 내려도(matte.mask_refined) 상자 테두리처럼
제품에 딱 붙은 것은 남는다 — 실측 T390 선물상자 26.7% → 16.1% 로 줄었을 뿐이다.

그래서 색으로 한 번 더 거른다. 금속에는 두 가지 신호가 있다.

- **채도(S)** — 금은 채도가 높다. 투명 아크릴·흰 케이스·종이는 낮다.
- **정반사 비율** — 은·백금·화이트골드는 채도가 낮아 위 조건에 걸리지만,
  거울처럼 반사해 **아주 밝은 화소**를 만든다. 이것이 안전장치다.

⚠️ **채도만으로 자르면 안 된다.** 화이트골드 반지·은 귀걸이가 통째로 배경이
된다(실측 T341 은 귀걸이는 채도 중앙값이 케이스와 구분되지 않는다).
반드시 `spec_ratio` 를 OR 로 함께 본다.

## 투명 케이스가 제품을 감싼 경우

아크릴 봉인 골드바는 케이스와 금이 **하나의 성분**으로 이어져 있어 성분 단위로는
못 가른다. 이때만 성분 안에서 Otsu 로 고채도 코어를 뽑는다. 다만 코어가 너무
작게 나오면(반사로 채도가 죽은 금이 통째로 빠지는 경우) 성분 전체를 남긴다.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

# 이 채도(0~1) 이상이면 금 계열로 본다.
# ⚠️ 0.35 는 우리 사진에 너무 높다 — 실측에서 T152 반지(0.306)와 T330 목걸이
#    (0.231)까지 배경으로 잘렸다. 저울 몸통은 0.051 이라 0.20 으로도 걸러진다.
_S_MIN = 0.20
# 아주 밝은 화소(정반사) 비율이 이 이상이면 은·백금 계열로 본다.
# ⚠️ 0.02 도 높다 — T152 반지가 0.0108, T330 목걸이가 0.0016 이다. 저울은 0.0000.
_R_MIN = 0.005
# 정반사로 셀 밝기 문턱
_SPEC_V = 0.94
# 투명 케이스에서 고채도 코어만 남길 때, 코어가 성분의 이 비율 미만이면 성분 전체를 쓴다
_CORE_MIN = 0.15
# 코어 추출을 **이 크기 이상의 성분에만** 건다.
#
# 케이스·상자가 제품과 한 덩어리로 붙는 것은 큰 성분에서만 일어난다(실측 T374
# 아크릴 14.2%, T390 선물상자 16.1%). 작은 성분(반지·체인)에까지 걸면 Otsu 가
# 물체 자신의 하이라이트와 그늘을 갈라 **마스크가 조각난다** — 실측으로 T379
# 반지가 20.58% → 7.77%(성분 6개), T341 귀걸이가 6.76% → 1.83%(성분 14개)로
# 부서졌다. 그래서 크기 조건을 반드시 함께 본다.
_CORE_MIN_COMPONENT_FRAC = 0.08
# 코어의 가장 큰 조각이 코어 전체의 이 비율 미만이면 "부서졌다"고 보고 되돌린다
_CORE_SINGLE_PIECE_MIN = 0.85
# 화면의 이 비율보다 작은 성분은 잡음으로 보고 건드리지 않는다
_MIN_COMPONENT_FRAC = 0.0005


def _is_single_piece(core: np.ndarray) -> bool:
    """
    코어가 **한 덩어리**인가.

    케이스를 벗겨 낸 결과는 제품 하나로 이어져 있다. 반대로 케이스가 없는데
    코어를 뽑으면 물체 자신의 하이라이트와 그늘이 갈려 여러 조각이 나온다 —
    실측 T379 반지가 성분 6개·구멍 13개로 부서졌다(면적도 20.58% → 9.87%).

    그래서 "가장 큰 조각이 코어의 대부분인가"로 케이스 벗기기와 자기 파괴를
    가른다. 조각나면 호출 측이 성분 전체를 그대로 쓴다.
    """
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    if n <= 1:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    return bool(areas.max() >= areas.sum() * _CORE_SINGLE_PIECE_MIN)


def drop_non_metal(mask: np.ndarray, bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    마스크에서 귀금속이 아닌 성분을 뺀다. (마스크 0/255, meta)

    성분을 **하나도 못 남기면 원본 마스크를 그대로 돌려준다** — 판정이 틀렸을 때
    "귀금속을 찾지 못했습니다"로 끝나는 것보다 케이스가 조금 붙어 있는 편이 낫다.
    """
    binary = (mask > 0).astype(np.uint8)
    if int(cv2.countNonZero(binary)) == 0:
        return mask, {"non_metal": "empty"}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0

    n, labels = cv2.connectedComponents(binary, connectivity=4)
    out = np.zeros_like(binary)
    kept, dropped, cored = 0, 0, 0

    for i in range(1, n):
        sel = labels == i
        area = int(sel.sum())
        if area < binary.size * _MIN_COMPONENT_FRAC:
            continue

        chroma_med = float(np.median(sat[sel]))
        spec_ratio = float((val[sel] > _SPEC_V).mean())
        # ⚠️ OR 이다. AND 로 바꾸면 은·백금이 전부 배경이 된다.
        if not (chroma_med >= _S_MIN or spec_ratio >= _R_MIN):
            dropped += 1
            continue

        # 투명 케이스·상자가 제품과 한 덩어리로 붙은 경우에만 고채도 코어를 뽑는다
        if area >= binary.size * _CORE_MIN_COMPONENT_FRAC:
            s_vals = (sat[sel] * 255).astype(np.uint8)
            th, _bin = cv2.threshold(s_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 금속의 정반사는 채도가 죽는다. 채도만 보면 하이라이트가 구멍이 되므로
            # **아주 밝은 화소는 무조건 살린다.**
            core = sel & ((sat * 255 > th) | (val > _SPEC_V))
            core = (
                cv2.morphologyEx(
                    core.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
                )
                > 0
            )
            if int(core.sum()) >= area * _CORE_MIN and _is_single_piece(core):
                out[core] = 1
                cored += 1
                kept += 1
                continue
        out[sel] = 1
        kept += 1

    meta = {
        "non_metal": "applied",
        "components_kept": kept,
        "components_dropped": dropped,
        "components_cored": cored,
    }
    if int(cv2.countNonZero(out)) == 0:
        # 전부 떨어졌다 = 판정이 틀렸다. 원본을 살린다.
        log.warning("drop_non_metal removed everything — 원본 마스크를 유지한다")
        meta["non_metal"] = "reverted_empty"
        return mask, meta

    return out * 255, meta
