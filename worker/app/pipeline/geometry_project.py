"""
Weak G1 정투영: 월드 좌표(mm, 카드 중심 원점) → 각 뷰 이미지 픽셀.

`geometry_g1.jewel_bbox_uv_mm` 의 역변환이며, 축·부호는 `view_axes.VIEW_AXIS_MAP`
**같은 테이블**을 본다(이전에는 별도 if-체인이라 좌/우 뷰에서 어긋났다).
"""

from __future__ import annotations

import numpy as np

from app.pipeline.card import CardGeometry
from app.pipeline.view_axes import axes_for_view


def world_mm_to_pixel_uv(
    view: str,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    card: CardGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """벡터화: world (x,y,z) mm → 이미지 (px, py)."""
    comp: dict[str, np.ndarray] = {"x": x_mm, "y": y_mm, "z": z_mm}
    u_axis, u_sign, v_axis, v_sign = axes_for_view(view)
    u_mm = comp[u_axis] * u_sign
    v_mm = comp[v_axis] * v_sign

    ccx = float(card.quad_px[:, 0].mean())
    ccy = float(card.quad_px[:, 1].mean())
    s = card.sigma_mm_per_px
    px = ccx + u_mm / s
    py = ccy - v_mm / s
    return px, py


def sample_mask(mask: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    mask[h,w], px/py same shape — 실루엣 내부 여부.

    **프레임 밖으로 투영된 점은 '바깥'으로 본다.** 이전 구현은 `np.clip` 으로 테두리
    픽셀에 붙여버려서, 테두리가 전경이면 화면 밖 복셀까지 살아남았다.
    """
    h, w = mask.shape[:2]
    xi = np.round(px).astype(np.int32)
    yi = np.round(py).astype(np.int32)
    in_frame = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi_c = np.clip(xi, 0, w - 1)
    yi_c = np.clip(yi, 0, h - 1)
    return (mask[yi_c, xi_c] > 127) & in_frame
