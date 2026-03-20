from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from app.config import Settings
from app.constants import CARD_DILATE_PX, JEWEL_AREA_FRAC_MAX, JEWEL_AREA_FRAC_MIN
from app.pipeline.card import CardGeometry
from app.pipeline.exceptions import PipelineError

log = logging.getLogger(__name__)


def _jewel_area_frac(jewel: np.ndarray, h: int, w: int) -> float:
    return int((jewel > 0).sum()) / float(h * w)


def _morph_clean(jewel: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    jewel = cv2.morphologyEx(jewel, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    jewel = cv2.morphologyEx(jewel, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    return jewel


def _card_d_and_inner(bgr: np.ndarray, card: CardGeometry) -> tuple[np.ndarray, np.ndarray]:
    """dilate(card) for §5.2 차집합 + 카드 내부(침식) — 카드 위 소물체 폴백용."""
    h, w = bgr.shape[:2]
    card_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(card_mask, [card.quad_px.astype(np.int32)], 255)
    kernel = np.ones((CARD_DILATE_PX, CARD_DILATE_PX), np.uint8)
    card_d = cv2.dilate(card_mask, kernel, iterations=1)
    ksz = max(3, min(h, w) // 50)
    if ksz % 2 == 0:
        ksz += 1
    inner = cv2.erode(card_mask, np.ones((ksz, ksz), np.uint8), iterations=1)
    if int(inner.sum()) < 255 * 10:
        inner = cv2.erode(card_mask, np.ones((3, 3), np.uint8), iterations=1)
    return card_d, inner


def _validate_jewel_area(jewel: np.ndarray, h: int, w: int, view: str) -> None:
    area = int((jewel > 0).sum())
    frac = area / float(h * w)
    if frac < JEWEL_AREA_FRAC_MIN:
        raise PipelineError(
            "ERR_SILHOUETTE_AREA",
            f"Jewel mask too small ({frac:.4f} of frame)",
            retry_step=view,
        )
    if frac > JEWEL_AREA_FRAC_MAX:
        raise PipelineError(
            "ERR_SILHOUETTE_AREA",
            f"Jewel mask too large ({frac:.4f} of frame)",
            retry_step=view,
        )


def _subtract_card_and_clean(
    fg: np.ndarray,
    bgr: np.ndarray,
    card: CardGeometry,
    view: str,
) -> np.ndarray:
    h, w = bgr.shape[:2]
    card_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(card_mask, [card.quad_px.astype(np.int32)], 255)
    kernel = np.ones((CARD_DILATE_PX, CARD_DILATE_PX), np.uint8)
    card_d = cv2.dilate(card_mask, kernel, iterations=1)
    jewel = cv2.bitwise_and(fg, cv2.bitwise_not(card_d))
    jewel = cv2.morphologyEx(jewel, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    jewel = cv2.morphologyEx(jewel, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    _validate_jewel_area(jewel, h, w, view)
    return jewel


def _jewel_mask_on_card_inner(
    fg: np.ndarray,
    gray: np.ndarray,
    card_inner: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    """카드 면 위에 올린 소형 물체(귀걸이 등) — 카드 내부 ∩ 전경."""
    jewel = cv2.bitwise_and(fg, card_inner)
    jewel = _morph_clean(jewel, 3, 5)
    if _jewel_area_frac(jewel, h, w) < JEWEL_AREA_FRAC_MIN:
        thr = min(110.0, float(gray.mean()) * 0.72)
        dark = ((gray.astype(np.float32) < thr).astype(np.uint8)) * 255
        jewel = cv2.bitwise_and(dark, card_inner)
        jewel = _morph_clean(jewel, 3, 5)
    return jewel


def _jewel_mask_heuristic(bgr: np.ndarray, card: CardGeometry, view: str) -> tuple[np.ndarray, str]:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(fg.mean()) > 127:
        fg = 255 - fg

    card_d, inner = _card_d_and_inner(bgr, card)

    jewel_sub = cv2.bitwise_and(fg, cv2.bitwise_not(card_d))
    jewel_sub = _morph_clean(jewel_sub, 5, 7)
    f_sub = _jewel_area_frac(jewel_sub, h, w)

    if f_sub < JEWEL_AREA_FRAC_MIN:
        thr = min(110.0, float(gray.mean()) * 0.72)
        dark = ((gray.astype(np.float32) < thr).astype(np.uint8)) * 255
        alt = cv2.bitwise_and(dark, cv2.bitwise_not(card_d))
        alt = _morph_clean(alt, 3, 5)
        if _jewel_area_frac(alt, h, w) > f_sub:
            jewel_sub = alt
            f_sub = _jewel_area_frac(jewel_sub, h, w)

    jewel_on = _jewel_mask_on_card_inner(fg, gray, inner, h, w)
    f_on = _jewel_area_frac(jewel_on, h, w)

    candidates: list[tuple[str, np.ndarray, float]] = [
        ("subtract_card", jewel_sub, f_sub),
        ("on_card_inner", jewel_on, f_on),
    ]
    valid = [(n, jm, fr) for n, jm, fr in candidates if JEWEL_AREA_FRAC_MIN <= fr <= JEWEL_AREA_FRAC_MAX]
    if valid:
        # 둘 다 유효하면 프레임 대비 더 작은 마스크 선호(배경 과포함 완화)
        valid.sort(key=lambda x: x[2])
        name, jm, fr = valid[0]
        log.info("jewel mask mode=%s frac=%.5f view=%s", name, fr, view)
        _validate_jewel_area(jm, h, w, view)
        return jm, name

    raise PipelineError(
        "ERR_SILHOUETTE_AREA",
        f"Jewel mask invalid (subtract_card frac={f_sub:.4f}, on_card_inner frac={f_on:.4f}). "
        "귀금속은 카드 옆 바닥에 나란히 두는 것을 권장합니다. 카드 위에만 올리면 인식이 불안정할 수 있습니다.",
        retry_step=view,
    )


def _jewel_mask_rembg(bgr: np.ndarray, card: CardGeometry, view: str) -> np.ndarray:
    try:
        from PIL import Image
        from rembg import remove
    except ImportError as e:
        raise RuntimeError("rembg/PIL required") from e

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    out = remove(pil.convert("RGB"))
    arr = np.array(out)
    if arr.ndim == 2:
        alpha = arr
    else:
        alpha = arr[:, :, 3] if arr.shape[2] >= 4 else (arr[:, :, 0] > 0).astype(np.uint8) * 255
    fg = ((alpha.astype(np.float32) > 100.0).astype(np.uint8)) * 255
    return _subtract_card_and_clean(fg, bgr, card, view)


def build_jewel_mask(
    bgr: np.ndarray,
    card: CardGeometry,
    settings: Settings,
    view: str,
    job_id: str = "debug",
) -> tuple[np.ndarray, dict[str, str]]:
    """
    §5.2: 전경에서 dilate(card) 제거.
    `ARCHIMEDES_SEGMENTATION_BACKEND=rembg` 시 rembg(옵션: `pip install -e ".[seg-rembg]"`).

    Returns:
        (mask, {"placement_mode": "subtract_card" | "on_card_inner"})
    """
    backend = (settings.segmentation_backend or "heuristic").strip().lower()
    placement_mode = "subtract_card"
    if backend == "rembg":
        try:
            jewel = _jewel_mask_rembg(bgr, card, view)
        except Exception as e:  # noqa: BLE001
            log.warning("rembg failed for %s, fallback heuristic: %s", view, e)
            jewel, placement_mode = _jewel_mask_heuristic(bgr, card, view)
    else:
        jewel, placement_mode = _jewel_mask_heuristic(bgr, card, view)

    if settings.reject_jewel_on_card and placement_mode == "on_card_inner":
        raise PipelineError(
            "ERR_JEWEL_ON_CARD",
            "귀금속이 신용카드 면 위에 올려진 것으로 보입니다. 카드와 같은 바닥에 "
            "**카드 옆**에 나란히 두고 다시 촬영해 주세요. 카드 위에 두면 카드 무늬·그림자가 "
            "함께 잡혀 실루엣이 비정상적으로 커지고, 무게가 수십~수백 g처럼 잘못 나올 수 있습니다.",
            retry_step=view,
        )

    if settings.debug_save_masks and os.path.isdir(settings.worker_output_dir):
        path = os.path.join(settings.worker_output_dir, f"{job_id}_{view}_jewel.png")
        cv2.imwrite(path, jewel)

    return jewel, {"placement_mode": placement_mode}
