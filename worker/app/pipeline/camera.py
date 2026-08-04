"""
카메라 내부 파라미터 K (`archimedes-v2-single-photo.mdc` §3.3).

우선순위: EXIF 35mm 환산 → EXIF 초점거리+센서폭 프리셋 → 기기 프리셋 → 폴백.
폴백까지 내려가면 신뢰도를 낮춰야 한다(`Intrinsics.is_reliable`).

⚠️ 이미지 크기는 **EXIF 회전을 적용한 뒤**의 것을 넘겨야 한다. 회전 전 크기를
쓰면 fx/fy 와 주점이 통째로 어긋난다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# 35mm 풀프레임 36×24mm 의 대각. 폭(36)만 쓰는 근사는 3:2 가 아닌 센서에서 어긋난다.
_FF35_DIAGONAL_MM = math.hypot(36.0, 24.0)

# 기기 모델 → 센서 폭(mm). EXIF FocalLength 는 있는데 35mm 환산이 없을 때 쓴다.
# project-concept §15.1: 실측·문헌으로 확장해야 하는 표다(현재는 자리만 잡아 둠).
DEVICE_SENSOR_WIDTH_MM: dict[str, float] = {}


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    source: str  # exif_35mm | exif_focal_sensor | device_preset | fallback

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def is_reliable(self) -> bool:
        """폴백 K 만 추측이다. 카드 소실점 해는 사진에서 **푼** 값이라 신뢰한다."""
        return self.source != "fallback"

    def as_meta(self) -> dict[str, Any]:
        return {
            "fx": round(self.fx, 3),
            "fy": round(self.fy, 3),
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "source": self.source,
            "reliable": self.is_reliable,
        }


def _principal_point(width_px: int, height_px: int) -> tuple[float, float]:
    return width_px / 2.0, height_px / 2.0


def intrinsics_from_exif(
    exif: dict[str, Any] | None, width_px: int, height_px: int
) -> Intrinsics:
    cx, cy = _principal_point(width_px, height_px)
    exif = exif or {}

    f35 = exif.get("focal_length_35mm")
    if f35:
        try:
            f35f = float(f35)
        except (TypeError, ValueError):
            f35f = 0.0
        if f35f > 0:
            # 35mm 환산은 대각 화각 기준이므로 대각으로 환산한다
            diag_px = math.hypot(width_px, height_px)
            f_px = f35f * diag_px / _FF35_DIAGONAL_MM
            return Intrinsics(f_px, f_px, cx, cy, "exif_35mm")

    f_mm = exif.get("focal_length_mm")
    model = str(exif.get("model") or "").strip()
    sensor_w = DEVICE_SENSOR_WIDTH_MM.get(model)
    if f_mm and sensor_w:
        try:
            f_px = float(f_mm) * width_px / float(sensor_w)
        except (TypeError, ValueError, ZeroDivisionError):
            f_px = 0.0
        if f_px > 0:
            return Intrinsics(f_px, f_px, cx, cy, "exif_focal_sensor")

    if model in DEVICE_SENSOR_WIDTH_MM:
        # 초점거리는 없지만 기기를 아는 경우를 위한 자리 — 프리셋 표가 채워지면 활성화
        pass

    # 최후 폴백: 소비자 폰의 전형적 화각 근사.
    # 이 값을 쓰면 절대 스케일을 K 에 의존해선 안 되고, 앵커(카드)로 잡아야 한다.
    f_px = 1.15 * float(max(width_px, height_px))
    return Intrinsics(f_px, f_px, cx, cy, "fallback")


def focal_from_rectangle_quad(
    quad_px, width_px: int, height_px: int, *, min_ratio: float = 0.25, max_ratio: float = 6.0
) -> float | None:
    """
    **알려진 직사각형**(신용카드)의 소실점으로 초점거리를 푼다.

    직사각형의 마주보는 변 두 쌍은 각각 소실점 v1, v2 를 만든다. 두 방향이
    3D 에서 직교하므로 주점 c 에 대해

        (v1 - c) · (v2 - c) = -f²

    가 성립한다. 즉 **EXIF 가 없어도 사진 한 장에서 f 를 얻을 수 있다.**

    왜 중요한가: 물체의 가로·세로 실측은 f 가 약분돼 f 오차에 둔감하지만
    (크기 = 픽셀 × 카드실측 / 카드픽셀), **바닥면 위 높이는 f 에 비례**한다.
    f 를 1.15·max(W,H) 로 추측하면 그 오차가 두께 → 부피 → 무게로 그대로 간다.

    카드가 정면에 가까우면 소실점이 무한대로 가 수치가 불안정하다 → `None`.
    """
    import numpy as _np

    q = _np.asarray(quad_px, dtype=_np.float64).reshape(4, 2)
    cx, cy = width_px / 2.0, height_px / 2.0

    def _line(a_pt, b_pt):
        return _np.cross(_np.array([a_pt[0], a_pt[1], 1.0]), _np.array([b_pt[0], b_pt[1], 1.0]))

    diag = float(_np.hypot(width_px, height_px))
    # 소실점이 이 거리보다 멀면 두 변이 사실상 평행하다 = 그 축엔 원근이 없다.
    # 수치적으로는 유한한 값이 나오지만 의미가 없어(실측 48억 px) f 가 폭주한다.
    far_limit = 60.0 * diag

    def _vanish(l1, l2):
        v = _np.cross(l1, l2)
        if abs(v[2]) < 1e-9:
            return None
        vp = _np.array([v[0] / v[2], v[1] / v[2]])
        if not _np.all(_np.isfinite(vp)):
            return None
        if float(_np.hypot(vp[0] - cx, vp[1] - cy)) > far_limit:
            return None
        return vp

    v1 = _vanish(_line(q[0], q[1]), _line(q[3], q[2]))
    v2 = _vanish(_line(q[1], q[2]), _line(q[0], q[3]))
    if v1 is None or v2 is None:
        return None

    f_sq = -float((v1[0] - cx) * (v2[0] - cx) + (v1[1] - cy) * (v2[1] - cy))
    if not _np.isfinite(f_sq) or f_sq <= 0:
        return None
    f = float(_np.sqrt(f_sq))

    # 소비자 폰의 물리적으로 말이 되는 범위 밖이면 신뢰하지 않는다
    span = float(max(width_px, height_px))
    if not (min_ratio * span <= f <= max_ratio * span):
        return None
    return f


def intrinsics_from_card(
    exif: dict[str, Any] | None, quad_px, width_px: int, height_px: int
) -> Intrinsics:
    """EXIF 우선, 없으면 **카드 소실점**으로 f 를 푼다. 그것도 안 되면 폴백."""
    from_exif = intrinsics_from_exif(exif, width_px, height_px)
    if from_exif.source != "fallback":
        return from_exif

    f = focal_from_rectangle_quad(quad_px, width_px, height_px)
    if f is None:
        return from_exif
    cx, cy = _principal_point(width_px, height_px)
    return Intrinsics(f, f, cx, cy, "card_vanishing_point")


def pixel_rays(
    K: Intrinsics, xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    픽셀 → 정규화 광선 방향의 (x, y) 성분. z 성분은 항상 1 이다.

    카메라 좌표계 점: `(rx * Z, ry * Z, Z)`.
    """
    rx = (xs - K.cx) / K.fx
    ry = (ys - K.cy) / K.fy
    return rx, ry
