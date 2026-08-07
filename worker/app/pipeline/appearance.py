"""
외형 기반 전경 후보 (깊이로 못 찾을 때의 폴백).

## 왜 Otsu 하나로는 안 되는가

`HeuristicSegmenter` 는 명도 Otsu 로 전경을 잡고, 밝은 쪽이 과반이면 뒤집는다.
"어두운 바닥 위 밝은 금속"에는 맞지만 **밝은 바닥에서는 뒤집혀서 그림자를
물체로 잡는다.**

실측(도련님 반지 사진): 밝은 베이지 책상 위 금반지. Otsu 가 반지 **안쪽 구멍의
그림자**를 전경으로 잡아 마스크가 반달 모양이 됐다(14.1×6.9mm). 정작 금속
밴드는 책상과 명도가 비슷해 배경으로 분류됐다.

## 채도(chroma) 경로

금은 채도가 높고 책상·그림자는 무채색에 가깝다. Lab 의 (a,b) 크기로 재면
**명도와 무관하게** 금속만 남는다. 실측에서 반지 밴드를 정확히 땄다.

다만 은·백금처럼 무채색 금속에는 안 통하므로 Otsu 를 대체하지 않고 **후보를
하나 더 얹고**, `local_lab_contrast` 로 더 그럴듯한 쪽을 고른다.
"""

from __future__ import annotations

import cv2
import numpy as np

# 낮은 임계 = Otsu × 이 비율. 히스테리시스의 아래쪽 문턱.
# 실측(반지 사진, 고리 실제 면적 ≈ 화면의 0.75%): 0.55 → 0.53%, **0.45 → 0.66%**,
# 0.35 → 6.24%(책상 질감이 이어져 딸려옴). 0.45 가 물체를 가장 많이 덮으면서
# 배경으로 새지 않는 지점이다.
_CHROMA_LOW_RATIO = 0.45


def chroma_foreground(bgr: np.ndarray) -> np.ndarray:
    """
    Lab 채도가 높은 화소를 전경으로. 명도가 배경과 같아도 금속이 남는다.

    **히스테리시스**를 쓴다. Otsu 한 번으로 자르면 물체의 그늘진 쪽이 떨어져
    나간다 — 실측(반지 사진)에서 밴드의 위쪽 절반만 남았다. 그렇다고 임계를
    그냥 낮추면 책상 질감이 딸려 온다(임계 0.4배에서 면적 4배).

    그래서 Otsu 를 **씨앗**으로 두고, 낮은 임계에서 그 씨앗과 **이어진** 화소만
    받아들인다. 같은 물체의 그늘은 이어져 있고, 무관한 배경 얼룩은 안 이어진다.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    peak = float(chroma.max())
    if peak <= 1e-6:
        return np.zeros(bgr.shape[:2], np.uint8)
    scaled = (255.0 * chroma / peak).astype(np.uint8)

    thr, high = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _t2, low = cv2.threshold(scaled, thr * _CHROMA_LOW_RATIO, 255, cv2.THRESH_BINARY)

    k = max(3, round(max(bgr.shape[:2]) * 0.004) | 1)
    low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    # 낮은 임계 성분 중 **씨앗을 품은 것만** 남긴다
    n, labels, _stats, _c = cv2.connectedComponentsWithStats(low, connectivity=8)
    if n <= 1:
        return cv2.morphologyEx(high, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    seeded = np.unique(labels[high > 0])
    keep = np.isin(labels, seeded[seeded > 0])
    return (keep.astype(np.uint8)) * 255


def local_lab_contrast(bgr: np.ndarray, mask: np.ndarray) -> float:
    """
    마스크 안쪽이 **바로 바깥 띠와 얼마나 다른가** (Lab 거리).

    물체라면 주변 바닥과 색이 다르다. 그림자는 명도만 조금 다르고 색은 같으므로
    이 값이 낮게 나온다 — 후보 둘 중 무엇이 진짜 물체인지 가르는 데 쓴다.
    """
    if int(cv2.countNonZero(mask)) < 32:
        return 0.0
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.where(mask > 0)
    span = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()), 8)
    band = max(3, round(span * 0.25))
    outer = cv2.dilate(mask, np.ones((band * 2 + 1, band * 2 + 1), np.uint8))
    ring = cv2.subtract(outer, mask)
    if int(cv2.countNonZero(ring)) < 32:
        return 0.0
    inside = np.median(lab[mask > 0], axis=0)
    around = np.median(lab[ring > 0], axis=0)
    return float(np.linalg.norm(inside - around))
