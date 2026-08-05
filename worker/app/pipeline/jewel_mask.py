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


def card_masks(card: CardGeometry, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(팽창된 카드 영역, 침식된 카드 내부)."""
    h, w = shape[:2]
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(filled, [np.asarray(card.quad_px, dtype=np.int32)], 255)
    dilated = cv2.dilate(filled, np.ones((CARD_DILATE_PX, CARD_DILATE_PX), np.uint8), iterations=1)
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
    카드 중심에서 카드 긴 변의 `spans` 배 안쪽 영역.

    `side` 가 "left"/"right" 면 **카드 기준 그쪽 절반만** 남긴다.
    촬영 규약으로 배치를 고정하면 탐색 영역이 절반으로 줄어 오검출이 크게 준다.
    좌우 판정은 이미지 x 축이 아니라 **카드의 긴 변 방향**을 기준으로 한다
    (카드가 기울어 찍혀도 규약이 성립하도록).
    """
    h, w = shape[:2]
    q = np.asarray(card.quad_px, dtype=np.float64).reshape(4, 2)
    cx, cy = float(q[:, 0].mean()), float(q[:, 1].mean())
    long_px, _ = card_edge_lengths_px(card.quad_px)

    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi, (round(cx), round(cy)), round(long_px * spans), 255, -1)
    if side not in ("left", "right"):
        return roi

    # 카드의 긴 변 방향 단위벡터. 이미지 좌표에서 +x 쪽이 "오른쪽"이 되도록 맞춘다.
    e_a, e_b = q[1] - q[0], q[2] - q[1]
    axis = e_a if np.linalg.norm(e_a) >= np.linalg.norm(e_b) else e_b
    if axis[0] < 0:
        axis = -axis
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)

    ys, xs = np.mgrid[0:h, 0:w]
    proj = (xs - cx) * axis[0] + (ys - cy) * axis[1]
    keep = proj < 0 if side == "left" else proj > 0
    return cv2.bitwise_and(roi, (keep.astype(np.uint8)) * 255)


def refine_jewel_mask(
    fg: np.ndarray,
    card: CardGeometry | None,
    view: str = "front",
    *,
    roi_card_spans: float = 1.0,
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

    # 카드 근처 최대 연결성분만 — 그림자·반사 조각 제거
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = ((labels == biggest).astype(np.uint8)) * 255
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
