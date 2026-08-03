"""
부피: 슬랩 AABB 교차 + (옵션) 복셀 카빙 Visual Hull 근사.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.constants import VIEW_ORDER, VOXEL_GRID_N
from app.pipeline.card import CardGeometry
from app.pipeline.exceptions import PipelineError
from app.pipeline.geometry_project import sample_mask, world_mm_to_pixel_uv
from app.pipeline.view_axes import view_world_intervals


@dataclass
class VolumeEstimate:
    V_hull_mm3: float
    multires_penalty: bool
    volume_model: str  # slab_aabb | slab_aabb_fallback | voxel_carve | depth_2p5d
    V_coarse_mm3: float | None = None
    multires_rel_diff: float | None = None
    grid_n: int | None = None


def _intersect_axis(
    current: tuple[float, float] | None,
    new: tuple[float, float],
    axis: str,
    view: str,
) -> tuple[float, float]:
    """
    누적 구간 ∩ 새 구간.

    **교집합이 비면 즉시 `ERR_VOLUME`.** 이전 구현은 빈 교집합에 `None` 을 반환했는데,
    다음 뷰 호출이 `None` 을 "아직 제약 없음"으로 오인해 자기 구간으로 덮어썼다.
    그래서 서로 모순된 뷰가 에러 없이 통과하고 해당 제약이 통째로 사라졌다.
    상세: `archimedes-v2-single-photo.mdc` §0.4 #3.
    """
    lo, hi = (new[0], new[1]) if new[0] <= new[1] else (new[1], new[0])
    if current is None:
        return (lo, hi)
    lo2 = max(current[0], lo)
    hi2 = min(current[1], hi)
    if lo2 >= hi2:
        raise PipelineError(
            "ERR_VOLUME",
            f"View '{view}' contradicts other views on axis {axis}: "
            f"{current} ∩ ({lo:.2f}, {hi:.2f}) is empty. "
            "각도별로 서로 다른 사진인지, 촬영 규약(정면/상/좌/우/후)에 맞는지 확인해 주세요.",
            retry_step=view,
            error_severity="soft",
            suggested_action="retry_one_view",
        )
    return (lo2, hi2)


def slab_aabb_intervals_mm(
    bboxes: dict[str, tuple[float, float, float, float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    bboxes[view] = (u_min, u_max, v_min, v_max) in mm (`geometry_g1` 규칙)
    → 월드 AABB (x, y, z) mm.

    뷰→월드 축 매핑은 `view_axes.VIEW_AXIS_MAP` 단일 소스를 따른다.
    """
    acc: dict[str, tuple[float, float] | None] = {"x": None, "y": None, "z": None}

    for view in VIEW_ORDER:
        if view not in bboxes:
            continue
        for axis, interval in view_world_intervals(view, bboxes[view]).items():
            acc[axis] = _intersect_axis(acc[axis], interval, axis, view)

    missing = [a for a in ("x", "y", "z") if acc[a] is None]
    if missing:
        raise PipelineError(
            "ERR_VOLUME",
            f"No view constrains world axis {missing} — missing views?",
            retry_step=None,
        )
    x_rng, y_rng, z_rng = acc["x"], acc["y"], acc["z"]
    assert x_rng is not None and y_rng is not None and z_rng is not None
    if (x_rng[1] - x_rng[0]) <= 0 or (y_rng[1] - y_rng[0]) <= 0 or (z_rng[1] - z_rng[0]) <= 0:
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
    bboxes: dict[str, tuple[float, float, float, float]],
    penalty_ratio: float,
    *,
    use_carving: bool,
    view_items: list[tuple[str, np.ndarray, CardGeometry]] | None,
    grid_n: int,
) -> VolumeEstimate:
    """
    슬랩 AABB(항상) + 복셀 카빙(옵션) → V_hull, 그리고 **해상도 민감도** 판정.

    다해상도 검사는 스펙 §8 대로 **동일 AABB 위에서 격자 N vs N/2** 를 비교한다.
    이전 구현은 bbox 를 0.85배로 줄인 것을 "coarse" 로 삼아, 부피가 결정론적으로
    ~0.61배가 되어 rel≈0.39 > 임계 0.12 → `multires_penalty` 가 **항상 True** 였다.
    그 결과 tier 는 precision_boost 가 없으면 절대 high 가 되지 못했다.
    """
    v_sl = volume_from_view_bboxes(bboxes)
    if not use_carving or not view_items:
        # 슬랩 단독에는 격자 해상도 개념이 없다 → 해상도 페널티는 판정하지 않고,
        # 모델 자체가 상한 근사라는 사실은 `volume_model` 로 신뢰도에 반영한다.
        return VolumeEstimate(V_hull_mm3=v_sl, multires_penalty=False, volume_model="slab_aabb")

    gn = max(16, min(int(grid_n), 96))
    v_fine = carve_visual_hull_mm3(view_items, bboxes, gn)
    if v_fine <= 0 or not np.isfinite(v_fine):
        return VolumeEstimate(
            V_hull_mm3=v_sl, multires_penalty=False, volume_model="slab_aabb_fallback"
        )

    # Visual hull ⊆ 슬랩 상자; 수치 오차 시 min
    v_final = float(min(v_fine, v_sl))

    gn_coarse = max(8, gn // 2)
    v_coarse = carve_visual_hull_mm3(view_items, bboxes, gn_coarse)
    rel: float | None = None
    pen = False
    if v_coarse > 0 and np.isfinite(v_coarse):
        rel = abs(v_fine - v_coarse) / max(v_fine, v_coarse, 1.0)
        pen = rel > penalty_ratio

    return VolumeEstimate(
        V_hull_mm3=v_final,
        multires_penalty=pen,
        volume_model="voxel_carve",
        V_coarse_mm3=float(v_coarse) if v_coarse > 0 else None,
        multires_rel_diff=rel,
        grid_n=gn,
    )


def grid_n_pair() -> tuple[int, int]:
    return VOXEL_GRID_N, VOXEL_GRID_N // 2
