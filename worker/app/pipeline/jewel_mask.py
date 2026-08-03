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
from app.pipeline.card import CardGeometry
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


def refine_jewel_mask(
    fg: np.ndarray,
    card: CardGeometry | None,
    view: str = "front",
) -> tuple[np.ndarray, dict[str, object]]:
    """
    전경에서 귀금속 마스크를 뽑는다.

    Returns (mask, {"placement_mode": "no_card" | "beside_card" | "on_card"}).
    """
    if card is None:
        mask = _morph_clean(fg, 5, 7)
        frac = _area_frac(mask)
        _validate(frac, view, "no_card")
        return mask, {"placement_mode": "no_card", "area_frac": round(frac, 6)}

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
    # 둘 다 유효하면 더 작은 쪽 — 배경·카드 과포함을 피한다
    valid.sort(key=lambda c: c[2])
    mode, mask, frac = valid[0]
    log.info("jewel mask mode=%s frac=%.5f view=%s", mode, frac, view)
    return mask, {"placement_mode": mode, "area_frac": round(frac, 6)}


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
