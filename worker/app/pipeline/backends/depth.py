"""
깊이 추정 백엔드 (계획서 Step 2-2 — "카메라에서 물체까지의 거리").

- `stub`: 상수 깊이(AFFINE_INVARIANT). 스케일 융합에서 앵커로 보정하면
  "물체가 카드 평면에 놓여 있다"는 v1 약원근 가정과 정확히 같아진다.
  즉 v1 을 v2 프레임 안에서 표현한 **정직한 퇴화 기준선**이다.
- `onnx`: 단일 입력 → 단일 (H,W) 깊이 맵을 내는 표준 계약.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from app.pipeline.backends.onnx_session import (
    letterbox,
    load_session,
    require_shape,
    to_nchw_rgb,
)
from app.pipeline.backends.types import DepthKind, DepthMap

log = logging.getLogger(__name__)

# ImageNet 정규화 — 대부분의 깊이 백본이 이 전처리로 학습된다
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@runtime_checkable
class DepthEstimator(Protocol):
    name: str

    def estimate(self, bgr: np.ndarray) -> DepthMap: ...


class ConstantDepthEstimator:
    """
    상수 깊이 스텁.

    스케일이 미정(AFFINE_INVARIANT)이므로 앵커가 있으면 §3 융합이 카드 평면 거리로
    맞춰 준다. 결과적으로 물체 두께는 관측되지 않고(h_vis=0) 제품별 최소 두께
    `t_min_k` 로 떨어진다 — **깊이 모델을 붙이기 전의 하한 동작**이다.
    """

    name = "stub"

    def __init__(self, value: float = 1.0) -> None:
        self.value = float(value)

    def estimate(self, bgr: np.ndarray) -> DepthMap:
        h, w = bgr.shape[:2]
        return DepthMap(
            depth=np.full((h, w), self.value, dtype=np.float32),
            kind=DepthKind.AFFINE_INVARIANT,
            meta={"backend": self.name, "degenerate": True},
        )


class OnnxDepthEstimator:
    """
    ONNX 깊이 모델.

    **출력 계약**: 하나의 출력, 형상 `(1,1,H,W)` / `(1,H,W)` / `(H,W)`.
    값의 의미는 모델에 따라 다르므로 `kind` 를 생성자에서 명시한다
    (metric 모델이면 `DepthKind.METRIC` 과 `output_scale_to_mm` 를 함께 준다).

    ⚠️ 시차(disparity)를 내는 모델(MiDaS·Depth Anything 계열)은
    `kind=DepthKind.INVERSE_AFFINE` 로 지정하고 `inverse=False` 로 둔다.
    스케일 융합이 **역깊이 공간에서** 아핀을 맞춘다(§3).
    `inverse=True` 는 여기서 1/x 를 취해 깊이로 바꾸는 옵션이며, 시차 모델에
    이걸 쓰면 β 항 때문에 체계적으로 틀어진다.
    """

    name = "onnx"

    def __init__(
        self,
        model_dir: str,
        filename: str = "depth.onnx",
        input_size: int = 518,
        kind: DepthKind = DepthKind.AFFINE_INVARIANT,
        inverse: bool = False,
        output_scale_to_mm: float = 1.0,
        session: object | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.filename = filename
        self.input_size = input_size
        self.kind = kind
        self.inverse = inverse
        self.output_scale_to_mm = float(output_scale_to_mm)
        self._session = session

    def _sess(self):
        if self._session is None:
            self._session = load_session(self.model_dir, self.filename)
        return self._session

    def estimate(self, bgr: np.ndarray) -> DepthMap:
        import cv2

        h, w = bgr.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(bgr, self.input_size, pad_value=0)
        tensor = to_nchw_rgb(canvas, _IMAGENET_MEAN, _IMAGENET_STD)

        sess = self._sess()
        input_name = sess.get_inputs()[0].name
        raw = np.asarray(sess.run(None, {input_name: tensor})[0], dtype=np.float32)
        require_shape("depth", raw, 2, 3, 4)
        while raw.ndim > 2:
            raw = raw[0]

        # letterbox 패딩 제거 후 원본 해상도로 복원
        nw = max(1, round(w * scale))
        nh = max(1, round(h * scale))
        if raw.shape != (self.input_size, self.input_size):
            raw = cv2.resize(raw, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        crop = raw[pad_y : pad_y + nh, pad_x : pad_x + nw]
        depth = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        valid = np.isfinite(depth)
        if self.inverse:
            # 역깊이 → 깊이. 0 근처는 무한대이므로 무효 처리.
            safe = np.abs(depth) > 1e-6
            valid &= safe
            depth = np.where(safe, 1.0 / np.where(safe, depth, 1.0), 0.0).astype(np.float32)

        depth = depth * self.output_scale_to_mm
        return DepthMap(
            depth=depth,
            kind=self.kind,
            valid=valid,
            meta={
                "backend": self.name,
                "model": self.filename,
                "inverse": self.inverse,
                "input_size": self.input_size,
            },
        )
