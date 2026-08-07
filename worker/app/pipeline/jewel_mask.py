"""
전경 마스크 → **귀금속만 남긴 마스크** (project-concept §5.2).

`backends.Segmenter` 가 낸 전경에서 카드 영역을 빼거나(카드 옆 배치),
카드 면 위에 놓인 경우 카드 내부와 교집합한다.

v1 과의 차이: v1 은 카드 **위**에 놓인 배치를 `ERR_JEWEL_ON_CARD` 로 거절했다.
슬랩 AABB 는 카드 무늬·그림자가 섞이면 부피가 폭주했기 때문이다.
v2 의 height_field 는 카드 평면을 바닥면으로 쓰므로 카드 위 배치도 기하적으로
성립한다 — 그래서 단일사진 경로에서는 거절하지 않고 배치 모드만 기록한다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.constants import CARD_DILATE_PX, JEWEL_AREA_FRAC_MAX, JEWEL_AREA_FRAC_MIN
from app.pipeline.card import CardGeometry, card_edge_lengths_px
from app.pipeline.exceptions import PipelineError

log = logging.getLogger(__name__)


def _morph_clean(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))


def _area_frac(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    return int(np.count_nonzero(mask)) / float(h * w)


def card_dilate_px(card: CardGeometry) -> int:
    """
    카드를 뺄 때 얼마나 넉넉히 뺄지. **카드 크기에 비례**시킨다.

    `CARD_DILATE_PX`(5) 는 고정 픽셀이라 4032px 사진에서는 사실상 0 이다.
    쿼드가 카드 실제 외곽선에서 수십 px 안쪽으로 들어가면 **카드 테두리가
    띠 모양으로 남아** 물체 후보가 된다.

    실측(도련님 반지 사진): 채도 전경에서 파란 카드의 테두리 띠(카드 bbox 와
    정확히 일치, 화면의 1.4%)가 반지(0.4%)를 제치고 최대 성분이 됐다.
    """
    _long, short = card_edge_lengths_px(card.quad_px)
    return max(CARD_DILATE_PX, round(short * 0.04))


def card_masks(card: CardGeometry, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(팽창된 카드 영역, 침식된 카드 내부)."""
    h, w = shape[:2]
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(filled, [np.asarray(card.quad_px, dtype=np.int32)], 255)
    d = card_dilate_px(card)
    dilated = cv2.dilate(filled, np.ones((d, d), np.uint8), iterations=1)
    k = max(3, (min(h, w) // 50) | 1)
    inner = cv2.erode(filled, np.ones((k, k), np.uint8), iterations=1)
    if int(np.count_nonzero(inner)) < 10:
        inner = cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=1)
    return dilated, inner


def card_roi_mask(
    card: CardGeometry,
    shape: tuple[int, int],
    spans: float = 1.0,
    side: str = "any",
) -> np.ndarray:
    """
    카드 중심에서 카드 긴 변의 `spans` 배 안쪽 영역 — **카드를 둘러싼 전 방향**.

    ⚠️ 예전에는 `side` 로 카드 기준 **반쪽만** 남겼다. 촬영 규약("귀금속 왼쪽,
    카드 오른쪽")을 코드로 강제한 것인데, 실사진은 그렇게 안 찍힌다 — 실측
    real5.jpg 에서 도련님은 금괴를 카드 **아래**에 놓으셨고, 물체가 통째로 ROI
    밖이라 대신 카드 인쇄물이 잡혔다. 규약을 어겼다고 분석을 실패시킬 이유가 없다.

    지금은 반경으로만 자르고(=책상 위 키보드·모니터는 여전히 배제),
    `side` 는 후보가 여럿일 때의 **가점**으로만 쓴다(`side_bias`).
    """
    h, w = shape[:2]
    q = np.asarray(card.quad_px, dtype=np.float64).reshape(4, 2)
    cx, cy = float(q[:, 0].mean()), float(q[:, 1].mean())
    long_px, _ = card_edge_lengths_px(card.quad_px)

    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi, (round(cx), round(cy)), round(long_px * spans), 255, -1)
    return roi


def card_long_axis(card: CardGeometry) -> np.ndarray:
    """카드 긴 변 방향 단위벡터. 이미지 좌표에서 +x 쪽이 '오른쪽'이 되게 맞춘다."""
    q = np.asarray(card.quad_px, dtype=np.float64).reshape(4, 2)
    e_a, e_b = q[1] - q[0], q[2] - q[1]
    axis = e_a if np.linalg.norm(e_a) >= np.linalg.norm(e_b) else e_b
    if axis[0] < 0:
        axis = -axis
    return axis / max(float(np.linalg.norm(axis)), 1e-9)


# 성분이 프레임 가장자리에 이만큼 닿으면 배경으로 본다
_BORDER_MARGIN_RATIO = 0.005


def touches_frame_border(
    stats_row: np.ndarray, shape: tuple[int, int], margin_ratio: float = _BORDER_MARGIN_RATIO
) -> bool:
    """
    연결성분이 **프레임 가장자리에 닿는가**. 닿으면 배경이다.

    촬영 규약상 귀금속은 카드 옆에, 프레임 안쪽에 통째로 보이게 둔다. 반면
    책상 너머 배경(컵·케이블·모니터)은 언제나 프레임 밖으로 이어지므로 가장자리에
    닿는다. 이 한 줄이 "물체 후보"와 "배경 조각"을 아주 싸게 가른다.

    실측(도련님 반지 사진): 카드 긴 변이 1721px 인데 카드 중심 y 가 1542 라
    반경 1배 ROI 가 프레임 위쪽(y=-179)을 넘어섰다. 그 안에 들어온 컵·케이블이
    바닥 위로 솟은 것으로 잡혀 최대 성분이 됐고, 정작 반지는 버려졌다.
    결과 15.604 g.
    """
    h, w = shape[0], shape[1]
    m = max(2, round(margin_ratio * max(h, w)))
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    ww = int(stats_row[cv2.CC_STAT_WIDTH])
    hh = int(stats_row[cv2.CC_STAT_HEIGHT])
    return x <= m or y <= m or (x + ww) >= (w - m) or (y + hh) >= (h - m)


def side_bias(card: CardGeometry, cx: float, cy: float, side: str) -> float:
    """
    규약대로 놓였으면 1.0, 반대편이면 0.6. **떨어뜨리지는 않는다.**

    좌우 판정은 이미지 x 축이 아니라 카드의 긴 변 방향 기준이다
    (카드가 기울어 찍혀도 규약이 성립하도록).
    """
    if side not in ("left", "right"):
        return 1.0
    q = np.asarray(card.quad_px, dtype=np.float64).reshape(4, 2)
    axis = card_long_axis(card)
    proj = (cx - float(q[:, 0].mean())) * axis[0] + (cy - float(q[:, 1].mean())) * axis[1]
    on_expected = proj < 0 if side == "left" else proj > 0
    return 1.0 if on_expected else 0.6


def refine_jewel_mask(
    fg: np.ndarray,
    card: CardGeometry | None,
    view: str = "front",
    *,
    roi_card_spans: float = 1.8,
    side: str = "any",
) -> tuple[np.ndarray, dict[str, object]]:
    """
    전경에서 귀금속 마스크를 뽑는다.

    카드가 있으면 **카드 주변으로 먼저 자른다.** 실사용 사진은 책상 위라
    프레임 절반이 키보드·모니터인데, 전경 마스크를 화면 전체에서 만들면
    그것들이 그대로 후보가 된다(실측: 마스크 24.5%, 복원 404×252mm).
    §4 프로토콜이 "카드 옆에 나란히"를 요구하므로 정당한 제약이다.

    Returns (mask, {"placement_mode": "no_card" | "beside_card" | "on_card"}).
    """
    if card is None:
        mask = _morph_clean(fg, 5, 7)
        frac = _area_frac(mask)
        _validate(frac, view, "no_card")
        return mask, {"placement_mode": "no_card", "area_frac": round(frac, 6)}

    roi = card_roi_mask(card, fg.shape, roi_card_spans, side)
    fg = cv2.bitwise_and(fg, roi)

    dilated, inner = card_masks(card, fg.shape)

    beside = _morph_clean(cv2.bitwise_and(fg, cv2.bitwise_not(dilated)), 5, 7)
    on_card = _morph_clean(cv2.bitwise_and(fg, inner), 3, 5)

    candidates = [("beside_card", beside, _area_frac(beside)), ("on_card", on_card, _area_frac(on_card))]
    valid = [c for c in candidates if JEWEL_AREA_FRAC_MIN <= c[2] <= JEWEL_AREA_FRAC_MAX]
    if not valid:
        raise PipelineError(
            "ERR_SILHOUETTE_AREA",
            f"Jewelry mask invalid (beside_card={candidates[0][2]:.5f}, "
            f"on_card={candidates[1][2]:.5f} of frame). "
            "귀금속이 프레임에 충분히 크게, 선명히 보이도록 다시 촬영해 주세요.",
            retry_step=view,
            error_severity="soft",
            suggested_action="retake_photo",
        )
    # **beside_card 를 우선**한다. §4 프로토콜이 "카드 옆에 나란히"를 요구하고
    # 촬영 가이드도 그렇게 안내한다.
    #
    # 이전에는 "더 작은 쪽"을 골랐는데, 신용카드 자체의 인쇄(원·로고)가 Otsu 에
    # 전경으로 잡혀 on_card 후보가 더 작게 나오는 일이 흔했다.
    # 실측: 카드 내부 74.8×33.7mm 를 물체로 잡아 0.05g 짜리가 5.27g 으로 나왔다.
    by_mode = {c[0]: c for c in valid}
    mode, mask, frac = by_mode.get("beside_card") or min(valid, key=lambda c: c[2])

    # 카드 근처 연결성분 하나만 — 그림자·반사 조각 제거.
    # 면적이 가장 큰 것을 쓰되, 촬영 규약대로 놓인 쪽에 가점을 준다. 규약을
    # 어긴 배치도 **버리지 않고 감점만** 한다(실측: 금괴가 카드 아래에 놓였다).
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        # 프레임 가장자리에 닿는 성분은 배경이다(화면 밖으로 이어지는 것).
        # 하나도 안 남으면 종전대로 전체에서 고른다 — 여기서 실패시키지는 않는다.
        pool = [i for i in range(1, n) if not touches_frame_border(stats[i], mask.shape)]
        if not pool:
            pool = list(range(1, n))
        scores = [
            float(stats[i, cv2.CC_STAT_AREA])
            * side_bias(card, float(centroids[i][0]), float(centroids[i][1]), side)
            for i in pool
        ]
        best = pool[int(np.argmax(scores))]
        mask = ((labels == best).astype(np.uint8)) * 255
        frac = _area_frac(mask)

    log.info("jewel mask mode=%s frac=%.5f view=%s", mode, frac, view)
    return mask, {
        "placement_mode": mode,
        "area_frac": round(frac, 6),
        "roi_card_spans": roi_card_spans,
        "object_side": side,
    }


def _validate(frac: float, view: str, mode: str) -> None:
    if frac < JEWEL_AREA_FRAC_MIN:
        raise PipelineError(
            "ERR_SILHOUETTE_AREA",
            f"Jewelry mask too small ({frac:.5f} of frame, mode={mode})",
            retry_step=view,
        )
    if frac > JEWEL_AREA_FRAC_MAX:
        raise PipelineError(
            "ERR_SILHOUETTE_AREA",
            f"Jewelry mask too large ({frac:.5f} of frame, mode={mode})",
            retry_step=view,
        )
