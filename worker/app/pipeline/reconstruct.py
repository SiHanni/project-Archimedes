"""
2D → 3D 역투영과 2.5D 부피 (`archimedes-v2-single-photo.mdc` §4, 계획서 Step 2-2 Fig.2·3).

    X = (u - cx) · Z / fx,   Y = (v - cy) · Z / fy,   Z = D(u, v)

## 부피를 어떻게 낼 것인가

깊이는 **보이는 면만** 준다. 그래서 부피는 관측만으로 정해지지 않는다.
두 가지 모드를 둔다.

1. **height_field (앵커 있을 때, 기본)**
   물체는 카드와 **같은 바닥면**에 놓여 있다(§4 프로토콜). 앵커 PnP 로 그 바닥
   평면을 알고 있으므로, 마스크의 각 광선을 따라 **표면에서 바닥까지의 부피**를
   적분한다. 바닥에 놓인 볼록 물체에 대해서는 이게 곧 실제 부피다.

2. **prism_pca (앵커 없을 때 폴백)**
   바닥면을 모르면 "투영 실면적 × 유효 두께"로 근사한다. 두께는 PCA 3주축의
   로버스트 치수(가시면의 비평면성)에서 얻고 제품별 물리 범위로 클램프한다.
   기울어진 평면은 PCA 3축이 0 에 가까우므로 이 모드는 원리적으로 약하다.

어느 모드든 뒷면·속 빈 부분은 못 보므로 α_k(§4.3)가 남은 몫을 흡수한다.

v1 대비 개선: 면적을 bbox 가 아니라 **마스크 실면적**으로 쓴다. 반지처럼 가운데가
빈 형상에서 bbox 기반 부피가 크게 과대되던 문제가 원인째 사라진다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.constants import (
    MAX_THICKNESS_MM_BY_PRODUCT,
    MIN_THICKNESS_MM_BY_PRODUCT,
)
from app.pipeline.camera import Intrinsics, pixel_rays
from app.pipeline.exceptions import PipelineError

log = logging.getLogger(__name__)

_MIN_POINTS = 32
# PCA 는 표본으로 충분하다 — 면적·부피는 항상 전체 픽셀로 계산한다
_PCA_MAX_POINTS = 200_000
# 깊이 이상치 절단 백분위
_TRIM_LO, _TRIM_HI = 1.0, 99.0
# 주축 방향 치수를 낼 때의 로버스트 범위
_EXTENT_LO, _EXTENT_HI = 2.5, 97.5


@dataclass(frozen=True)
class SupportPlane:
    """물체가 놓인 바닥 평면 `n · X = d` (카메라 좌표계, mm)."""

    normal: np.ndarray
    d_mm: float

    def ray_depth(self, rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
        """각 광선이 이 평면과 만나는 지점의 깊이 Z."""
        denom = self.normal[0] * rx + self.normal[1] * ry + self.normal[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.d_mm / denom


@dataclass
class Reconstruction:
    area_proj_mm2: float
    length_mm: float
    width_mm: float
    h_vis_mm: float
    h_mean_mm: float
    volume_mm3: float
    n_points: int
    method: str  # height_field | prism_pca
    thickness_clamp: str | None = None  # "min" | "max" | None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_meta(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "area_proj_mm2": round(self.area_proj_mm2, 3),
            "length_mm": round(self.length_mm, 3),
            "width_mm": round(self.width_mm, 3),
            "h_vis_mm": round(self.h_vis_mm, 4),
            "h_mean_mm": round(self.h_mean_mm, 4),
            "n_points": self.n_points,
            "thickness_clamp": self.thickness_clamp,
            **self.meta,
        }


def thickness_bounds(product_k: str) -> tuple[float, float]:
    pk = (product_k or "other").lower()
    lo = MIN_THICKNESS_MM_BY_PRODUCT.get(pk, MIN_THICKNESS_MM_BY_PRODUCT["other"])
    hi = MAX_THICKNESS_MM_BY_PRODUCT.get(pk, MAX_THICKNESS_MM_BY_PRODUCT["other"])
    return lo, hi


def masked_samples(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    K: Intrinsics,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """마스크 내부의 유효 픽셀 → (rx, ry, Z). 깊이 이상치는 백분위로 절단."""
    if mask.shape[:2] != depth_mm.shape[:2]:
        raise PipelineError(
            "ERR_DEPTH_FAILED",
            f"mask {mask.shape[:2]} and depth {depth_mm.shape[:2]} shape mismatch",
        )
    sel = mask > 0
    if valid is not None:
        sel &= valid
    ys, xs = np.where(sel)
    if ys.size == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty, empty

    z = depth_mm[ys, xs].astype(np.float64)
    good = np.isfinite(z) & (z > 0)
    ys, xs, z = ys[good], xs[good], z[good]
    if z.size >= _MIN_POINTS:
        lo, hi = np.percentile(z, [_TRIM_LO, _TRIM_HI])
        keep = (z >= lo) & (z <= hi)
        ys, xs, z = ys[keep], xs[keep], z[keep]

    rx, ry = pixel_rays(K, xs.astype(np.float64), ys.astype(np.float64))
    return rx, ry, z


def backproject(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    K: Intrinsics,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """마스크 내부 픽셀 → 카메라 좌표계 3D 점 (N,3) mm 와 깊이 (N,)."""
    rx, ry, z = masked_samples(mask, depth_mm, K, valid)
    return np.stack([rx * z, ry * z, z], axis=1), z


def projected_area_mm2(z_mm: np.ndarray, K: Intrinsics) -> float:
    """
    마스크의 **실면적**.

    깊이 Z 인 픽셀 하나가 덮는 실면적은 `Z² / (fx·fy)` 이다(시선에 수직인 면 기준).
    픽셀별로 더하므로 bbox 근사와 달리 반지 구멍 같은 빈 곳을 세지 않는다.
    """
    if z_mm.size == 0:
        return 0.0
    return float(np.sum(z_mm**2) / (K.fx * K.fy))


def principal_extents_mm(points: np.ndarray) -> tuple[float, float, float]:
    """
    PCA 주축 3방향의 로버스트 치수 (내림차순).

    세 번째 성분은 가시면의 **비평면성**이다. 평평한 판을 기울여 찍으면 0 에
    가깝게 나오는 것이 정상이다 — 한 장으로는 두께를 볼 수 없기 때문이다.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0, 0.0, 0.0
    if n > _PCA_MAX_POINTS:
        step = int(np.ceil(n / _PCA_MAX_POINTS))
        points = points[::step]
    centered = points - points.mean(axis=0, keepdims=True)
    # SVD 가 공분산 고유분해보다 수치적으로 안정적이다
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt.T
    lo = np.percentile(proj, _EXTENT_LO, axis=0)
    hi = np.percentile(proj, _EXTENT_HI, axis=0)
    ext = np.sort(np.abs(hi - lo))[::-1]
    return float(ext[0]), float(ext[1]), float(ext[2])


def _height_field_volume(
    rx: np.ndarray, ry: np.ndarray, z: np.ndarray, plane: SupportPlane, K: Intrinsics
) -> tuple[float, np.ndarray]:
    """
    광선별로 표면 → 바닥 평면 사이 부피를 적분한다.

    한 픽셀이 쓸고 가는 절두체 부피 ≈ (중간 깊이에서의 화소 실면적) × (깊이 차).
    바닥보다 뒤에 있는 표본은 0 으로 클립한다(그림자·배경 누수 방어).
    """
    z_plane = plane.ray_depth(rx, ry)
    ok = np.isfinite(z_plane) & (z_plane > 0)
    dz = np.where(ok, z_plane - z, 0.0)
    dz = np.clip(dz, 0.0, None)
    z_mid = 0.5 * (z + np.where(ok, z_plane, z))
    dv = (z_mid**2) / (K.fx * K.fy) * dz
    return float(np.sum(dv)), dz


def reconstruct_from_depth(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    K: Intrinsics,
    product_k: str,
    support_plane: SupportPlane | None = None,
    valid: np.ndarray | None = None,
) -> Reconstruction:
    """마스크 + 절대 깊이 (+ 바닥면) → 실치수와 2.5D 부피."""
    rx, ry, z = masked_samples(mask, depth_mm, K, valid)
    if z.size < _MIN_POINTS:
        raise PipelineError(
            "ERR_DEPTH_FAILED",
            f"Only {z.size} valid depth points inside the jewelry mask "
            f"(need >= {_MIN_POINTS}). 물체가 선명히 보이도록 다시 촬영해 주세요.",
        )

    area = projected_area_mm2(z, K)
    pts = np.stack([rx * z, ry * z, z], axis=1)
    length, width, h_vis = principal_extents_mm(pts)
    t_min, t_max = thickness_bounds(product_k)

    if support_plane is not None:
        volume, dz = _height_field_volume(rx, ry, z, support_plane, K)
        method = "height_field"
        extra: dict[str, Any] = {"height_p95_mm": round(float(np.percentile(dz, 95)), 4)}
    else:
        volume = area * h_vis
        method = "prism_pca"
        extra = {}

    h_mean = volume / area if area > 0 else 0.0

    # 물리적 클램프 — 조용히 깎지 않고 플래그로 남긴다.
    # 두 경로 모두 여기 한 곳에서만 클램프해야 플래그가 빠지지 않는다.
    clamp: str | None = None
    if h_mean < t_min:
        clamp = "min"
        volume = area * t_min
    elif h_mean > t_max:
        clamp = "max"
        volume = area * t_max
    if clamp:
        h_mean = volume / area if area > 0 else 0.0

    return Reconstruction(
        area_proj_mm2=area,
        length_mm=length,
        width_mm=width,
        h_vis_mm=h_vis,
        h_mean_mm=h_mean,
        volume_mm3=volume,
        n_points=int(z.size),
        method=method,
        thickness_clamp=clamp,
        meta={
            "thickness_bounds_mm": [t_min, t_max],
            "mean_distance_mm": round(float(z.mean()), 2),
            **extra,
        },
    )
