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
        """폴백 K 는 초점거리를 추측한 것이라 신뢰도 감점 대상이다."""
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
