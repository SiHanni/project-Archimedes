"""
**미터 단위** 단안 깊이 (에라토스테네스 — 카드 없이 재는 경로).

## 왜 별도 모델이 필요한가

아르키메데스가 쓰는 Depth Anything V2 는 **affine-invariant** 다. 출력 d 와 실제
거리 Z 사이에 `1/Z = a·d + b` 관계만 있고 a, b 는 모른다. 지금까지는 **신용카드**가
그 a, b 를 풀어 줬다(ID-1 실측 85.60×53.98mm → PnP → 카드 평면까지의 절대 거리).

카드를 빼면 단안 스케일 모호성이 그대로 남아 **절대 거리가 원리적으로 안 나온다.**
EXIF 초점거리라도 있으면 다른 길이 있는데, 실측상 도련님 사진은 EXIF 가 지워져
들어온다(APP1 Exif 세그먼트가 140바이트 — 방향·크기뿐).

그래서 **스스로 초점거리까지 추정하는 metric depth 모델**이 필요하다.
Apple Depth Pro 는 사진 한 장에서 미터 단위 깊이와 화각을 함께 낸다.

## 변환식 (apple/ml-depth-pro `DepthPro.infer`)

모델은 정규화된 역깊이 `c` 와 수평 화각 `fov_deg` 를 낸다.

    f_px = 0.5 · W / tan(0.5 · fov_rad)
    inverse_depth = c · (W / f_px)
    Z = 1 / clamp(inverse_depth, 1e-4, 1e4)      # 미터

여기서 **W 는 리사이즈 전 원본 폭**이다. 모델 입력 폭(1536)을 쓰면 스케일이
통째로 어긋난다.

⚠️ 라이선스: Apple Sample Code License(`apple-ascl`). 사내 데모·연구 범위를
넘어 제품에 넣기 전에 반드시 확인할 것.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Depth Pro 는 정사각 1536 입력으로 학습됐다
DEFAULT_INPUT_SIZE = 1536
# 소비자 접사 범위 밖은 모델이 헛본 것으로 간주
MIN_PLAUSIBLE_M = 0.02
MAX_PLAUSIBLE_M = 20.0


@dataclass
class MetricDepth:
    """미터 단위 깊이맵 + 그 사진에서 추정한 초점거리."""

    depth_m: np.ndarray  # (H, W) 원본 해상도, 미터
    focal_px: float  # 원본 픽셀 기준
    fov_deg: float

    def as_meta(self) -> dict[str, Any]:
        finite = self.depth_m[np.isfinite(self.depth_m)]
        return {
            "backend": "depth_pro",
            "focal_px": round(self.focal_px, 2),
            "fov_deg": round(self.fov_deg, 3),
            "depth_min_m": round(float(finite.min()), 4) if finite.size else None,
            "depth_max_m": round(float(finite.max()), 4) if finite.size else None,
            "depth_median_m": round(float(np.median(finite)), 4) if finite.size else None,
        }


class DepthProEstimator:
    """
    ONNX Depth Pro. 출력 이름이 배포본마다 달라 **형상으로 골라낸다**.

    깊이맵은 원소가 가장 많은 출력, 화각은 스칼라 출력으로 본다.
    """

    name = "depth_pro"

    def __init__(self, model_dir: str, filename: str, input_size: int = DEFAULT_INPUT_SIZE) -> None:
        self.model_dir = model_dir
        self.filename = filename
        self.input_size = input_size
        self._session: Any = None

    def _sess(self) -> Any:
        if self._session is None:
            from app.pipeline.backends.onnx_session import load_session

            self._session = load_session(self.model_dir, self.filename)
            ins = [(i.name, i.shape) for i in self._session.get_inputs()]
            outs = [(o.name, o.shape) for o in self._session.get_outputs()]
            log.info("depth_pro loaded inputs=%s outputs=%s", ins, outs)
            # 입력이 정사각으로 고정돼 있으면 그 값을 따른다
            shape = self._session.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
                self.input_size = int(shape[2])
        return self._session

    def estimate(self, bgr: np.ndarray) -> MetricDepth:
        h, w = bgr.shape[:2]
        sess = self._sess()

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        # rescale 1/255 → normalize (x-0.5)/0.5  == [-1, 1]
        tensor = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        out_names = [o.name for o in sess.get_outputs()]
        raw = sess.run(None, {sess.get_inputs()[0].name: tensor})
        arrays = [np.asarray(r, dtype=np.float32) for r in raw]
        canonical = max(arrays, key=lambda a: a.size)
        canonical = np.squeeze(canonical)
        if canonical.ndim != 2:
            raise RuntimeError(f"depth_pro: 깊이 출력 형상이 이상합니다 {canonical.shape}")

        # 배포본마다 두 번째 출력이 **화각(deg)** 이거나 **초점거리(px)** 다.
        # 이름으로 구분한다 — 값만 보고 넘기면 1731(px)을 1731도로 읽는다(실측 사고).
        scalar_idx = next(
            (i for i, a in enumerate(arrays) if a.size <= 4 and a.ndim <= 2), None
        )
        if scalar_idx is None:
            raise RuntimeError("depth_pro: 초점거리/화각 출력을 찾지 못했습니다")
        scalar = float(np.asarray(arrays[scalar_idx]).reshape(-1)[0])
        scalar_name = out_names[scalar_idx] if scalar_idx < len(out_names) else ""

        if "focal" in scalar_name.lower():
            # ⚠️ 그래프 안에서 **모델 입력 폭(1536)** 기준으로 계산된 값이다.
            #    원본 폭으로 환산하지 않으면 스케일이 통째로 어긋난다.
            focal_px = scalar * (w / float(self.input_size))
            fov_deg = math.degrees(2.0 * math.atan(0.5 * self.input_size / max(scalar, 1e-6)))
        else:
            fov_deg = scalar
            tan_half = math.tan(0.5 * math.radians(fov_deg))
            if not (1e-6 < tan_half < 1e6):
                raise RuntimeError(f"depth_pro: 화각이 비정상입니다 ({fov_deg} deg)")
            focal_px = 0.5 * w / tan_half

        if not (0.2 * w <= focal_px <= 8.0 * max(h, w)):
            raise RuntimeError(f"depth_pro: 초점거리가 비정상입니다 ({focal_px:.0f}px)")

        # W 는 **원본 폭**. 모델 입력 폭을 쓰면 스케일이 통째로 어긋난다.
        inverse_depth = np.clip(canonical * (w / focal_px), 1e-4, 1e4)
        depth_m = 1.0 / inverse_depth
        if depth_m.shape != (h, w):
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_LINEAR)

        return MetricDepth(depth_m=depth_m, focal_px=focal_px, fov_deg=fov_deg)


def distance_to_mask_mm(depth_m: np.ndarray, mask: np.ndarray) -> float | None:
    """
    마스크 영역까지의 거리(mm). **중앙값**을 쓴다 — 반사·구멍에서 튀는 값을 견딘다.
    """
    sel = (mask > 0) & np.isfinite(depth_m)
    if int(np.count_nonzero(sel)) < 32:
        return None
    z = float(np.median(depth_m[sel]))
    if not (MIN_PLAUSIBLE_M <= z <= MAX_PLAUSIBLE_M):
        return None
    return z * 1000.0
