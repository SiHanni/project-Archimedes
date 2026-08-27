"""
마스크 안에 남은 **비금속 덩어리**를 파낸다 (계획서 Step 1 후처리 2단계).

## 무엇이 남는가

크롭 재추론(matte.mask_refined)과 성분 단위 제외(non_metal.drop_non_metal)를
거쳐도 두 가지가 남는다. 둘 다 **제품과 한 성분으로 이어져 있어** 성분 단위로는
못 가른다.

1. **용기가 제품에 붙은 경우** — 분홍 선물상자·검은 스펀지가 금괴와 한 덩어리
   (실측 T390: 마스크 16.11% 중 상자·스펀지가 37.8%)
2. **구멍으로 배경이 비치는 경우** — 반지 안쪽으로 흰 받침이 보이는데 경계가
   약해 모델이 덜 파낸다 (실측 T332). 구멍 속이 어두운 사진(T379)은 잘 파낸다.

## 어떻게 가르는가

색상(H)이 결정적이다. 실측 히스토그램:

    금 제품    H 10~30 에 80~99% (T374 98.4% · T379 99.5% · T152 84.9%)
    분홍 상자  H 160~180 에 37.8%  ← 금과 겹치지 않는다
    흰 받침    채도가 죽고 밝다
    검은 스펀지 밝기가 죽는다

⚠️ **금속의 정반사(하이라이트)도 채도가 죽고 밝다.** 흰 받침과 구분이 안 된다.
   그래서 색으로 고른 뒤 **모폴로지 열기**로 거른다 — 하이라이트는 흩어진 작은
   점이라 사라지고, 받침·상자는 큰 덩어리라 살아남는다. 이것이 이 모듈의 핵심이다.

⚠️ **은·백금은 채도가 낮지만 밝지 않다**(실측 T341 은 귀걸이 채도 0.06~0.11 ·
   밝기 0.34). 밝기 조건이 그것들을 지켜 준다. 밝기 문턱을 내리지 말 것.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

# 금 계열 색상대 (OpenCV H 는 0~179). 실측: 금 제품이 이 안에 80~99% 든다.
_GOLD_H_LO, _GOLD_H_HI = 5, 45
# 밝고 채도 죽은 것 = 흰 받침·종이
_PAPER_V_MIN, _PAPER_S_MAX = 0.72, 0.22
# 금이 아닌 색인데 채도가 살아 있는 것 = 색깔 있는 포장(분홍 상자 등)
_COLORED_S_MIN = 0.18
# 아주 어두운 것 = 검은 스펀지·그늘
_DARK_V_MAX = 0.15
# 파낼 후보에서 흩어진 하이라이트를 지우는 열기 커널 (긴 변 대비)
_OPEN_RATIO = 0.012
# 이 비율(마스크 대비) 미만의 후보 덩어리는 무시한다
_MIN_CARVE_FRAC = 0.01
# 파낸 뒤 마스크가 원래의 이 비율 밑으로 줄면 되돌린다
_MIN_KEEP_RATIO = 0.45
# 정반사 화소로 셀 밝기 문턱
_SPEC_V = 0.94
# 덩어리 안의 정반사 비율이 이 이상이면 **금속으로 보고 건드리지 않는다.**
#
# 색만 보면 금속을 판다. 실측 —
#   T192 목걸이 체인·펜던트  색상 100(청록 반사)  정반사 0.072~0.076  ← 금속
#   T332 반지 윗면           색상  17(금)         정반사 0.674        ← 금속
#   T390 분홍 상자           색상 170(자홍)       정반사 0.000        ← 포장
# 금속은 거울처럼 반사해 아주 밝은 화소를 만들고, 포장지·받침은 만들지 못한다.
# 이 한 줄이 "색이 이상한 금속"과 "색이 이상한 포장"을 가른다.
_METAL_SPEC_MIN = 0.02


def carve_non_metal(mask: np.ndarray, bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """마스크 안의 비금속 덩어리를 파낸다. (마스크 0/255, meta)"""
    binary = (mask > 0).astype(np.uint8)
    total = int(cv2.countNonZero(binary))
    if total == 0:
        return mask, {"carve": "empty"}

    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.int16)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0

    gold_hue = (hue >= _GOLD_H_LO) & (hue <= _GOLD_H_HI)
    paper = (val >= _PAPER_V_MIN) & (sat <= _PAPER_S_MAX)
    colored = (~gold_hue) & (sat >= _COLORED_S_MIN)
    dark = val <= _DARK_V_MAX

    candidate = (binary > 0) & (paper | colored | dark)

    # ⚠️ 하이라이트 제거. 이 열기가 없으면 금 표면의 정반사가 전부 구멍이 된다.
    k = max(3, int(_OPEN_RATIO * max(h, w)) | 1)
    opened = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    n, labels, stats, _c = cv2.connectedComponentsWithStats(opened, connectivity=8)
    carve = np.zeros((h, w), np.uint8)
    blobs = 0
    protected = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < total * _MIN_CARVE_FRAC:
            continue
        sel = labels == i
        if float((val[sel] > _SPEC_V).mean()) >= _METAL_SPEC_MIN:
            protected += 1
            continue
        carve[sel] = 1
        blobs += 1

    if blobs == 0:
        return mask, {"carve": "nothing", "carve_protected": protected}

    out = binary.copy()
    out[carve > 0] = 0
    kept = int(cv2.countNonZero(out))
    ratio = kept / float(total)
    meta: dict[str, Any] = {
        "carve": "applied",
        "carve_blobs": blobs,
        "carve_protected": protected,
        "carve_keep_ratio": round(ratio, 3),
    }

    # 너무 많이 깎였다 = 물체 자신을 판 것이다
    if ratio < _MIN_KEEP_RATIO:
        meta["carve"] = "reverted_too_much"
        return mask, meta

    return out * 255, meta
