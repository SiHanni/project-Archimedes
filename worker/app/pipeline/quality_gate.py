from __future__ import annotations

import cv2
import numpy as np

from app.config import Settings
from app.pipeline.exceptions import PipelineError


def check_image_quality(bgr: np.ndarray, settings: Settings, view: str) -> None:
    h, w = bgr.shape[:2]
    short = min(h, w)
    if short < settings.min_short_edge_px:
        raise PipelineError(
            "ERR_RESOLUTION",
            f"Short edge {short}px < {settings.min_short_edge_px}px",
            retry_step=view,
        )

    gray = bgr[:, :, 1] if bgr.ndim == 3 else bgr  # green channel quick
    lap = cv2_laplacian_var(gray)
    if lap < settings.blur_laplacian_threshold:
        raise PipelineError(
            "ERR_BLUR",
            f"Laplacian variance {lap:.1f} below threshold",
            retry_step=view,
        )

    mean = float(gray.mean())
    if mean < 25.0 or mean > 230.0:
        raise PipelineError(
            "ERR_EXPOSURE",
            f"Mean intensity {mean:.1f} out of acceptable range",
            retry_step=view,
        )


def cv2_laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())
