"""
누끼 정밀화 (계획서 Step 1 — "귀금속 외곽선 정확히 추출").

## 왜 필요한가

높이 세그멘테이션(`height_segment`)은 바닥 위로 솟은 것을 찾는다. 이게 가장
견고하지만 **진짜 납작한 제품에는 원리적으로 안 통한다** — 0.05g 골드바는
두께가 깊이 노이즈(RMSE ~1.2mm)보다 훨씬 얇아 평면과 구분되지 않는다.
그럴 때는 외형(Otsu) 폴백으로 내려가는데, 금은 반사가 심해 **밝게 빛나는
일부만** 전경으로 잡힌다(실측: 금괴 상단 40% 만 마스크에 들어옴).

GrabCut 은 씨앗 영역의 **색 분포를 학습해** 같은 색의 나머지를 끌어온다.
금괴처럼 균일한 색이 어두운 책상 위에 있는 구도에 정확히 들어맞는다.

## 안전장치

GrabCut 은 씨앗이 나쁘면 배경까지 삼킨다. 결과 면적이 씨앗 대비 정해진
배수를 벗어나면 **원본 마스크를 그대로 돌려준다.** 정밀화가 실패해도
분석은 계속돼야 한다 — 나빠질 바엔 안 하느니만 못하다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

# 씨앗 bbox 를 이 비율만큼 넓혀 잘라 낸다(놓친 부분이 들어오도록).
# 기준은 bbox 의 **긴 변**이다 — 씨앗이 물체의 일부만 덮으면 짧은 변 기준
# 여백은 물체를 다 담지 못한다. 크롭 테두리는 확정 배경이라, 물체가 크롭
# 밖으로 나가면 그 부분은 영원히 못 살린다(실측: 막대 아래 22줄이 잘렸다).
_CROP_MARGIN = 1.0
# 결과가 씨앗 대비 이 범위를 벗어나면 폭주로 보고 되돌린다.
# 실측 성장 배수: 정상 2.1~2.7(금괴), 폭주 6.3(책상을 통째로 먹은 경우).
# Otsu 씨앗이 물체의 30~50% 를 잡으므로 정상 범위는 2~3배다.
_MIN_GROWTH = 0.5
_MAX_GROWTH = 4.0
_ITERS = 5
# GrabCut 은 내부 k-means 를 `cv2.theRNG()` 로 초기화한다. 시드를 고정하지 않으면
# **같은 사진을 같은 프로세스에서 두 번 돌릴 때 마스크가 미세하게 달라진다**
# (실측: 같은 이미지의 두 호출에서 마스크 면적 0.010022 vs 0.010038, 부피 0.16% 차).
# 측정 도구는 같은 입력에 같은 값을 내야 한다.
_RNG_SEED = 20260101


def refine_with_grabcut(
    bgr: np.ndarray,
    seed_mask: np.ndarray,
    *,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    씨앗 마스크를 색 분포로 넓혀 물체 전체를 덮는다.

    `exclude` (예: 카드 영역)는 확정 배경으로 못박아 정밀화가 그쪽으로
    새는 것을 막는다.

    Returns (mask, meta). 실패·폭주 시 `seed_mask` 를 그대로 돌려준다.
    """
    h, w = seed_mask.shape[:2]
    seed_area = int(cv2.countNonZero(seed_mask))
    meta: dict[str, object] = {"matting": "grabcut", "seed_area_px": seed_area}
    if seed_area < 64:
        meta["matting"] = "skipped_small_seed"
        return seed_mask, meta

    ys, xs = np.where(seed_mask > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    margin = round(max(x1 - x0 + 1, y1 - y0 + 1) * _CROP_MARGIN)
    cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
    cx1, cy1 = min(w, x1 + margin + 1), min(h, y1 + margin + 1)
    if cx1 - cx0 < 16 or cy1 - cy0 < 16:
        meta["matting"] = "skipped_small_crop"
        return seed_mask, meta

    patch = np.ascontiguousarray(bgr[cy0:cy1, cx0:cx1])
    seed = seed_mask[cy0:cy1, cx0:cx1]

    # 씨앗 안쪽은 확정 전경, 그 주변 띠는 '아마 전경', 나머지는 '아마 배경'
    k = max(3, round(min(patch.shape[:2]) * 0.02) | 1)
    sure_fg = cv2.erode(seed, np.ones((k, k), np.uint8))
    if int(cv2.countNonZero(sure_fg)) < 32:
        sure_fg = seed
    maybe_fg = cv2.dilate(seed, np.ones((k * 3, k * 3), np.uint8))

    gc = np.full(patch.shape[:2], cv2.GC_PR_BGD, np.uint8)
    gc[maybe_fg > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    # 크롭 테두리는 확정 배경 — 물체는 씨앗 주변에 있지 테두리에 있지 않다
    gc[0, :] = gc[-1, :] = gc[:, 0] = gc[:, -1] = cv2.GC_BGD
    if exclude is not None:
        gc[(exclude[cy0:cy1, cx0:cx1] > 0) & (sure_fg == 0)] = cv2.GC_BGD

    try:
        cv2.setRNGSeed(_RNG_SEED)
        cv2.grabCut(
            patch, gc, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
            _ITERS, cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as e:
        log.warning("grabcut failed: %s", e)
        meta["matting"] = "failed"
        return seed_mask, meta

    refined = (((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)) * 255
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((k * 2 + 1, k * 2 + 1), np.uint8))

    # 씨앗과 이어진 성분만 — GrabCut 이 멀리 떨어진 같은 색 얼룩도 함께 살릴 수 있다
    n, labels, _stats, _c = cv2.connectedComponentsWithStats(refined, connectivity=8)
    if n <= 1:
        meta["matting"] = "empty_result"
        return seed_mask, meta
    overlaps = [int(np.count_nonzero((labels == i) & (sure_fg > 0))) for i in range(1, n)]
    if max(overlaps) == 0:
        meta["matting"] = "lost_seed"
        return seed_mask, meta
    best = 1 + int(np.argmax(overlaps))
    refined = ((labels == best).astype(np.uint8)) * 255

    out_area = int(cv2.countNonZero(refined))
    growth = out_area / float(seed_area)
    meta["grabcut_growth"] = round(growth, 3)
    if not (_MIN_GROWTH <= growth <= _MAX_GROWTH):
        meta["matting"] = "rejected_growth"
        return seed_mask, meta

    full = np.zeros((h, w), np.uint8)
    full[cy0:cy1, cx0:cx1] = refined
    meta["refined_area_px"] = out_area
    return full, meta
