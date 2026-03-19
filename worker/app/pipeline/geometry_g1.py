"""
G1 v0: map jewel silhouette bbox from image pixels to each view's (u,v) mm plane
using card-centered weak scale (sigma). Full weak-perspective voxel carve: 백로그(implementation-plan §3).
"""

from __future__ import annotations

import numpy as np

from app.pipeline.card import CardGeometry


def jewel_bbox_uv_mm(mask: np.ndarray, card: CardGeometry, view: str) -> tuple[float, float, float, float]:
    """
    Returns (u_min, u_max, v_min, v_max) in mm for this view's projection coordinates.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return 0.0, 0.0, 0.0, 0.0
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    ccx = float(card.quad_px[:, 0].mean())
    ccy = float(card.quad_px[:, 1].mean())
    s = card.sigma_mm_per_px

    # Pixel deltas -> view-plane mm (see archimedes-implementation-plan §3 / concept §3)
    def span(dx0: float, dx1: float, dy0: float, dy1: float) -> tuple[float, float, float, float]:
        u0 = (dx0 - ccx) * s
        u1 = (dx1 - ccx) * s
        v0 = -(dy1 - ccy) * s  # image y down
        v1 = -(dy0 - ccy) * s
        return min(u0, u1), max(u0, u1), min(v0, v1), max(v0, v1)

    u0, u1, v0, v1 = span(x0, x1, y0, y1)

    if view == "front":
        return u0, u1, v0, v1  # u~x, v~z
    if view == "top":
        return u0, u1, v0, v1  # u~x, v~y
    if view == "left":
        return u0, u1, v0, v1  # u~z, v~y
    if view == "right":
        # Mirror u so world z aligns with left view
        return -u1, -u0, v0, v1
    if view == "back":
        return -u1, -u0, v0, v1  # u~-x, v~z
    return u0, u1, v0, v1
