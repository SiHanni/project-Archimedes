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


def _jewel_mask_heuristic(bgr: np.ndarray, card: CardGeometry, view: str) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(fg.mean()) > 127:
        fg = 255 - fg

    h, w = bgr.shape[:2]
    card_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(card_mask, [card.quad_px.astype(np.int32)], 255)
    kernel = np.ones((CARD_DILATE_PX, CARD_DILATE_PX), np.uint8)
    card_d = cv2.dilate(card_mask, kernel, iterations=1)
    jewel = cv2.bitwise_and(fg, cv2.bitwise_not(card_d))
    jewel = cv2.morphologyEx(jewel, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    jewel = cv2.morphologyEx(jewel, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    area = int((jewel > 0).sum())
    frac = area / float(h * w)
    if frac < JEWEL_AREA_FRAC_MIN:
        thr = min(110.0, float(gray.mean()) * 0.72)
        dark = ((gray.astype(np.float32) < thr).astype(np.uint8)) * 255
        jewel = cv2.bitwise_and(dark, cv2.bitwise_not(card_d))
        jewel = cv2.morphologyEx(jewel, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        jewel = cv2.morphologyEx(jewel, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    _validate_jewel_area(jewel, h, w, view)
    return jewel


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
) -> np.ndarray:
    """
    §5.2: 전경에서 dilate(card) 제거.
    `ARCHIMEDES_SEGMENTATION_BACKEND=rembg` 시 rembg(옵션: `pip install -e ".[seg-rembg]"`).
    """
    backend = (settings.segmentation_backend or "heuristic").strip().lower()
    if backend == "rembg":
        try:
            jewel = _jewel_mask_rembg(bgr, card, view)
        except Exception as e:  # noqa: BLE001
            log.warning("rembg failed for %s, fallback heuristic: %s", view, e)
            jewel = _jewel_mask_heuristic(bgr, card, view)
    else:
        jewel = _jewel_mask_heuristic(bgr, card, view)

    if settings.debug_save_masks and os.path.isdir(settings.worker_output_dir):
        path = os.path.join(settings.worker_output_dir, f"{job_id}_{view}_jewel.png")
        cv2.imwrite(path, jewel)

    return jewel
