"""
바닥면 위 **높이**로 귀금속을 찾는다.

## 왜 외형 세그를 안 쓰는가

실사용 사진은 책상 위다 — 키보드·모니터·상자가 프레임의 절반을 차지한다.
밝기(Otsu)나 범용 배경제거는 그걸 전부 "전경"으로 잡는다.
실측: 마스크가 화면의 24.5%, 복원 크기 404×252mm, 무게 10.6kg.

우리는 이미 **정확한 바닥 평면**을 갖고 있다(카드 앵커 PnP, 실측 깊이 RMSE 0.6mm).
그러면 "카드 옆 바닥에 놓인 물체"는 **평면 위로 솟은 점들**로 정의된다.
색·조명·배경 무늬와 무관해 훨씬 견고하다.

## 왜 카드 주변으로 제한하는가

책상 위 다른 물건도 바닥 위에 솟아 있다. §4 프로토콜이 "카드 옆에 나란히"를
요구하므로, **카드 근처**로 한정하면 잡동사니가 자연히 빠진다.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from app.constants import JEWEL_AREA_FRAC_MAX, JEWEL_AREA_FRAC_MIN
from app.pipeline.camera import Intrinsics
from app.pipeline.card import CardGeometry, card_edge_lengths_px
from app.pipeline.exceptions import PipelineError
from app.pipeline.jewel_mask import card_dilate_px, side_bias, touches_frame_border
from app.pipeline.reconstruct import SupportPlane

log = logging.getLogger(__name__)

# 깊이 노이즈(카드 위 홀드아웃 RMSE 실측 ~0.6mm)보다 넉넉히 위
DEFAULT_MIN_HEIGHT_MM = 2.0
# 소비자 귀금속이 이보다 두꺼울 일은 없다 — 넘으면 다른 물건이다
DEFAULT_MAX_HEIGHT_MM = 60.0
# 카드 긴 변의 배수 — "카드 옆"의 정량적 정의.
#
# 1.0 은 좁다. 카드 중심에서 재는 반경이라, 물체가 카드에서 카드 한 변만큼
# 떨어져 있으면 물체의 **바깥쪽 절반이 잘린다**(실측 반지: 중심 거리 1613px,
# 반지름 385px, ROI 반경 1721px → 왼쪽 호가 ROI 밖이라 마스크가 반쪽이 됐다).
# 넓혀도 안전한 이유: 프레임 테두리에 닿는 성분을 이미 배제하고(배경 제거),
# 촬영 규약 방향에 가점을 준다.
DEFAULT_ROI_CARD_SPANS = 1.8


def height_above_plane_mm(
    depth_mm: np.ndarray, plane: SupportPlane, K: Intrinsics
) -> np.ndarray:
    """각 픽셀에서 바닥 평면 위 높이(mm). 평면 뒤는 음수."""
    h, w = depth_mm.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    rx = (xs - K.cx) / K.fx
    ry = (ys - K.cy) / K.fy
    return plane.ray_depth(rx, ry) - depth_mm


def segment_by_height(
    depth_mm: np.ndarray,
    plane: SupportPlane,
    K: Intrinsics,
    card: CardGeometry,
    *,
    min_height_mm: float = DEFAULT_MIN_HEIGHT_MM,
    max_height_mm: float = DEFAULT_MAX_HEIGHT_MM,
    roi_card_spans: float = DEFAULT_ROI_CARD_SPANS,
    depth_rmse_mm: float | None = None,
    side: str = "any",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Returns (mask, meta). 카드 근처에서 바닥 위로 솟은 **최대 연결성분**.

    `depth_rmse_mm`(카드 홀드아웃 실측)이 주어지면 높이 임계를 그 3배 이상으로
    올린다. 자기 노이즈보다 작은 높이차를 "물체"라고 주장하면 안 된다 —
    실측에서 RMSE 9.7mm 인데 임계 2mm 를 쓰자 화면의 25% 가 물체로 잡혔다.
    """
    if depth_rmse_mm is not None and depth_rmse_mm > 0:
        min_height_mm = max(min_height_mm, 3.0 * float(depth_rmse_mm))
    h, w = depth_mm.shape[:2]
    height = height_above_plane_mm(depth_mm, plane, K)

    cx = float(card.quad_px[:, 0].mean())
    cy = float(card.quad_px[:, 1].mean())
    long_px, _ = card_edge_lengths_px(card.quad_px)
    ys, xs = np.mgrid[0:h, 0:w]
    # 카드를 둘러싼 **전 방향**. 예전에는 규약대로 한쪽 반원만 봤는데, 실사진은
    # 그렇게 안 찍힌다 — 실측 real5.jpg 는 금괴가 카드 아래에 놓여 통째로 ROI
    # 밖이었고, 대신 카드 인쇄물이 물체로 잡혔다. 규약은 아래에서 **가점**으로만 쓴다.
    roi = np.hypot(xs - cx, ys - cy) < long_px * roi_card_spans

    raised = np.isfinite(height) & (height > min_height_mm) & (height < max_height_mm)
    mask = ((roi & raised).astype(np.uint8)) * 255

    # 카드 자신은 평면이라 이론상 height≈0 이지만, 두께(0.76mm)·검출 오차로
    # 테두리가 남는다. 명시적으로 뺀다.
    card_fill = np.zeros((h, w), np.uint8)
    cv2.fillPoly(card_fill, [np.asarray(card.quad_px, dtype=np.int32)], 255)
    d = card_dilate_px(card)
    card_d = cv2.dilate(card_fill, np.ones((d, d), np.uint8))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(card_d))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # 연결성분 하나만 — 그림자 얼룩·노이즈 조각 제거.
    # 규약대로 놓인 쪽에 가점을 주되, 어긴 배치도 버리지 않고 감점만 한다.
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise PipelineError(
            "ERR_SILHOUETTE_AREA",
            "카드 옆 바닥에서 물체를 찾지 못했습니다. 귀금속을 카드 바로 옆에 "
            "두고, 물체가 화면에서 충분히 크게 보이도록 더 가까이 찍어 주세요.",
            error_severity="soft",
            suggested_action="retake_photo",
        )
    # 프레임 가장자리에 닿는 성분은 배경이다 — 물체는 통째로 프레임 안에 있다.
    # 실측(반지 사진): ROI 원이 프레임 위쪽을 넘어서 컵·케이블이 후보가 됐고,
    # 그게 최대 성분이라 반지 대신 채택됐다(15.604 g).
    inner = [i for i in range(1, n) if not touches_frame_border(stats[i], (h, w))]
    if not inner:
        raise PipelineError(
            "ERR_SILHOUETTE_AREA",
            "카드 옆에서 물체를 찾지 못했습니다(찾은 것이 모두 화면 밖으로 이어집니다). "
            "귀금속이 화면 안에 통째로 보이도록 다시 찍어 주세요.",
            error_severity="soft",
            suggested_action="retake_photo",
        )
    scores = [
        float(stats[i, cv2.CC_STAT_AREA])
        * side_bias(card, float(centroids[i][0]), float(centroids[i][1]), side)
        for i in inner
    ]
    best = inner[int(np.argmax(scores))]
    mask = ((labels == best).astype(np.uint8)) * 255

    frac = float(np.count_nonzero(mask)) / float(h * w)
    if not (JEWEL_AREA_FRAC_MIN <= frac <= JEWEL_AREA_FRAC_MAX):
        raise PipelineError(
            "ERR_SILHOUETTE_AREA",
            f"찾은 물체 크기가 비정상입니다(화면의 {frac * 100:.2f}%). "
            "귀금속을 카드 바로 옆에 두고 더 가까이 찍어 주세요.",
            error_severity="soft",
            suggested_action="retake_photo",
        )

    med_h = float(np.median(height[mask > 0]))
    log.info("height segment frac=%.5f median_height=%.2fmm", frac, med_h)
    return mask, {
        "backend": "depth_plane",
        "min_height_mm": round(min_height_mm, 3),
        "area_frac": round(frac, 6),
        "median_height_mm": round(med_h, 3),
        "roi_card_spans": roi_card_spans,
        "object_side": side,
        "height_band_mm": [min_height_mm, max_height_mm],
    }
