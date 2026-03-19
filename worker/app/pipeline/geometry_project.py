"""
Weak G1: 월드 좌표(mm, 카드 중심 원점 가정) → 각 뷰 이미지 픽셀.
jewel_bbox_uv_mm 과 역변환 관계를 맞춘다.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.card import CardGeometry


def world_mm_to_pixel_uv(
    view: str,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    card: CardGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """벡터화: world (x,y,z) mm → 이미지 (px, py)."""
    if view == "front":
        u_mm, v_mm = x_mm, z_mm
    elif view == "top":
        u_mm, v_mm = x_mm, y_mm
    elif view == "left":
        u_mm, v_mm = z_mm, y_mm
    elif view == "right":
        u_mm, v_mm = -z_mm, y_mm
    elif view == "back":
        u_mm, v_mm = -x_mm, z_mm
    else:
        u_mm, v_mm = x_mm, z_mm
    ccx = float(card.quad_px[:, 0].mean())
    ccy = float(card.quad_px[:, 1].mean())
    s = card.sigma_mm_per_px
    px = ccx + u_mm / s
    py = ccy - v_mm / s
    return px, py


def sample_mask(mask: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """mask[h,w], px/py same shape — bool inside silhouette."""
    h, w = mask.shape[:2]
    xi = np.clip(np.round(px).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(py).astype(np.int32), 0, h - 1)
    return mask[yi, xi] > 127
