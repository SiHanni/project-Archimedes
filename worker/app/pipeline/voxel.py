"""
부피: 슬랩 AABB 교차 + (옵션) 복셀 카빙 Visual Hull 근사.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.constants import VOXEL_GRID_N
from app.pipeline.card import CardGeometry
from app.pipeline.exceptions import PipelineError
from app.pipeline.geometry_project import sample_mask, world_mm_to_pixel_uv


@dataclass
class VolumeEstimate:
    V_hull_mm3: float
    multires_penalty: bool
    volume_model: str  # slab_aabb | voxel_carve


def _intersect_1d(
    current: tuple[float, float] | None, new: tuple[float, float]
) -> tuple[float, float] | None:
    lo, hi = new
    if lo > hi:
        lo, hi = hi, lo
    if current is None:
        return (lo, hi)
    a, b = current
    lo2 = max(a, lo)
    hi2 = min(b, hi)
    if lo2 >= hi2:
        return None
    return (lo2, hi2)


def slab_aabb_intervals_mm(
    bboxes: dict[str, tuple[float, float, float, float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    bboxes[view] = (u_min, u_max, v_min, v_max) in mm — geometry_g1 규칙.
    """
    x_rng: tuple[float, float] | None = None
    y_rng: tuple[float, float] | None = None
    z_rng: tuple[float, float] | None = None

    if "front" in bboxes:
        u0, u1, v0, v1 = bboxes["front"]
        x_rng = _intersect_1d(x_rng, (u0, u1))
        z_rng = _intersect_1d(z_rng, (v0, v1))
    if "top" in bboxes:
        u0, u1, v0, v1 = bboxes["top"]
        x_rng = _intersect_1d(x_rng, (u0, u1))
        y_rng = _intersect_1d(y_rng, (v0, v1))
    if "left" in bboxes:
        u0, u1, v0, v1 = bboxes["left"]
        z_rng = _intersect_1d(z_rng, (u0, u1))
        y_rng = _intersect_1d(y_rng, (v0, v1))
    if "right" in bboxes:
        u0, u1, v0, v1 = bboxes["right"]
        z_rng = _intersect_1d(z_rng, (u0, u1))
        y_rng = _intersect_1d(y_rng, (v0, v1))
    if "back" in bboxes:
        u0, u1, v0, v1 = bboxes["back"]
        x_rng = _intersect_1d(x_rng, (u0, u1))
        z_rng = _intersect_1d(z_rng, (v0, v1))

    if x_rng is None or y_rng is None or z_rng is None:
        raise PipelineError(
            "ERR_VOLUME",
            "Could not intersect view bboxes — missing views?",
            retry_step=None,
        )
    dx = x_rng[1] - x_rng[0]
    dy = y_rng[1] - y_rng[0]
    dz = z_rng[1] - z_rng[0]
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise PipelineError("ERR_VOLUME", "Degenerate AABB after intersection", retry_step=None)
    return x_rng, y_rng, z_rng


def volume_from_view_bboxes(
    bboxes: dict[str, tuple[float, float, float, float]],
) -> float:
    x_rng, y_rng, z_rng = slab_aabb_intervals_mm(bboxes)
    dx = x_rng[1] - x_rng[0]
    dy = y_rng[1] - y_rng[0]
    dz = z_rng[1] - z_rng[0]
    return float(dx * dy * dz)


def _expand_range(r: tuple[float, float], m: float = 0.03) -> tuple[float, float]:
    lo, hi = r
    w = hi - lo
    if w < 1e-6:
        return lo - 1.0, hi + 1.0
    return lo - m * w, hi + m * w


def carve_visual_hull_mm3(
    view_items: list[tuple[str, np.ndarray, CardGeometry]],
    bboxes: dict[str, tuple[float, float, float, float]],
    grid_n: int,
) -> float:
    """
    슬랩 AABB(소폭 팽창) 안의 균일 격자에서, 5뷰 마스크 교집합 복셀 부피 합.
    """
    x_rng, y_rng, z_rng = slab_aabb_intervals_mm(bboxes)
    x_rng = _expand_range(x_rng)
    y_rng = _expand_range(y_rng)
    z_rng = _expand_range(z_rng)

    gx = np.linspace(x_rng[0], x_rng[1], grid_n + 1, dtype=np.float64)
    gy = np.linspace(y_rng[0], y_rng[1], grid_n + 1, dtype=np.float64)
    gz = np.linspace(z_rng[0], z_rng[1], grid_n + 1, dtype=np.float64)
    cx = 0.5 * (gx[:-1] + gx[1:])
    cy = 0.5 * (gy[:-1] + gy[1:])
    cz = 0.5 * (gz[:-1] + gz[1:])
    cell_x = (x_rng[1] - x_rng[0]) / grid_n
    cell_y = (y_rng[1] - y_rng[0]) / grid_n
    cell_z = (z_rng[1] - z_rng[0]) / grid_n
    cell_vol = float(cell_x * cell_y * cell_z)

    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    flat_x = X.ravel()
    flat_y = Y.ravel()
    flat_z = Z.ravel()
    inside = np.ones(flat_x.shape[0], dtype=bool)
    for view, mask, card in view_items:
        px, py = world_mm_to_pixel_uv(view, flat_x, flat_y, flat_z, card)
        inside &= sample_mask(mask, px, py)
    count = int(inside.sum())
    return float(count * cell_vol)


def estimate_volume(
    bboxes_fine: dict[str, tuple[float, float, float, float]],
    bboxes_coarse: dict[str, tuple[float, float, float, float]] | None,
    penalty_ratio: float,
    *,
    use_carving: bool,
    view_items: list[tuple[str, np.ndarray, CardGeometry]] | None,
    grid_n: int,
) -> VolumeEstimate:
    v_sl = volume_from_view_bboxes(bboxes_fine)
    if not use_carving or not view_items:
        return _multires_only_slab(bboxes_fine, bboxes_coarse, penalty_ratio, v_sl)

    gn = max(16, min(int(grid_n), 96))
    v_carve = carve_visual_hull_mm3(view_items, bboxes_fine, gn)
    if v_carve <= 0 or not np.isfinite(v_carve):
        v_final = v_sl
        model = "slab_aabb_fallback"
    else:
        # Visual hull ⊆ 슬랩 상자; 수치 오차 시 min
        v_final = float(min(v_carve, v_sl))
        model = "voxel_carve"

    pen = False
    if bboxes_coarse and v_final > 0:
        gn2 = max(16, gn // 2)
        v_carve2 = carve_visual_hull_mm3(view_items, bboxes_coarse, gn2)
        v_sl2 = volume_from_view_bboxes(bboxes_coarse)
        v2 = float(min(v_carve2, v_sl2)) if v_carve2 > 0 else v_sl2
        rel = abs(v_final - v2) / max(v_final, v2, 1.0)
        pen = rel > penalty_ratio

    return VolumeEstimate(V_hull_mm3=v_final, multires_penalty=pen, volume_model=model)


def _multires_only_slab(
    bboxes_fine: dict[str, tuple[float, float, float, float]],
    bboxes_coarse: dict[str, tuple[float, float, float, float]] | None,
    penalty_ratio: float,
    v_f: float,
) -> VolumeEstimate:
    if not bboxes_coarse:
        return VolumeEstimate(V_hull_mm3=v_f, multires_penalty=False, volume_model="slab_aabb")
    v_c = volume_from_view_bboxes(bboxes_coarse)
    rel = abs(v_f - v_c) / max(v_f, v_c, 1.0)
    return VolumeEstimate(
        V_hull_mm3=v_f,
        multires_penalty=rel > penalty_ratio,
        volume_model="slab_aabb",
    )


def coarse_bboxes(
    bboxes: dict[str, tuple[float, float, float, float]],
    factor: float = 0.5,
) -> dict[str, tuple[float, float, float, float]]:
    out = {}
    for k, (u0, u1, v0, v1) in bboxes.items():
        uc = 0.5 * (u0 + u1)
        vc = 0.5 * (v0 + v1)
        hu = 0.5 * (u1 - u0) * factor
        hv = 0.5 * (v1 - v0) * factor
        out[k] = (uc - hu, uc + hu, vc - hv, vc + hv)
    return out


def grid_n_pair() -> tuple[int, int]:
    return VOXEL_GRID_N, VOXEL_GRID_N // 2


# 하위 호환
def volume_with_multires_check(
    bboxes_fine: dict[str, tuple[float, float, float, float]],
    bboxes_coarse: dict[str, tuple[float, float, float, float]] | None,
    penalty_ratio: float,
) -> VolumeEstimate:
    return _multires_only_slab(bboxes_fine, bboxes_coarse, penalty_ratio, volume_from_view_bboxes(bboxes_fine))
