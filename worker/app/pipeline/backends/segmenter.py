"""
분할 백엔드 (계획서 Step 1 — "귀금속 외곽선 정확히 추출").

- `heuristic`: 기존 `segment.py` Otsu 경로를 인터페이스로 감싼 것(폴백).
- `rembg`: 옵션 패키지.
- `onnx`: 단일 입력 → 단일 확률맵을 내는 매팅/분할 모델 계약.
  (SAM 처럼 **프롬프트 기반** 2단(encoder/decoder) 모델은 계약이 달라
   별도 백엔드로 두어야 한다 — `box` 인자는 그 이행을 위해 미리 열어 둔다.)
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from app.pipeline.backends.onnx_session import (
    contract_error,
    letterbox,
    load_session,
    require_shape,
    to_nchw_rgb,
)
from app.pipeline.backends.types import Detection, SegmentResult

log = logging.getLogger(__name__)


@runtime_checkable
class Segmenter(Protocol):
    name: str

    def segment(self, bgr: np.ndarray, box: Detection | None = None) -> SegmentResult:
        """전경(귀금속) 이진 마스크. `box` 가 있으면 그 영역을 우선 고려한다."""
        ...


class HeuristicSegmenter:
    """Otsu 전경 + (선택) 박스 제한. 카드 차집합은 상위 `segment.py` 가 담당한다."""

    name = "heuristic"

    def segment(self, bgr: np.ndarray, box: Detection | None = None) -> SegmentResult:
        import cv2

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(fg.mean()) > 127:
            fg = 255 - fg
        fg = _restrict_to_box(fg, box)
        return SegmentResult(mask=fg, meta={"backend": self.name})


class RembgSegmenter:
    """`pip install -e ".[seg-rembg]"` 필요."""

    name = "rembg"

    def segment(self, bgr: np.ndarray, box: Detection | None = None) -> SegmentResult:
        import cv2
        from PIL import Image
        from rembg import remove

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = np.array(remove(Image.fromarray(rgb).convert("RGB")))
        if out.ndim == 2:
            alpha = out
        elif out.shape[2] >= 4:
            alpha = out[:, :, 3]
        else:
            alpha = (out[:, :, 0] > 0).astype(np.uint8) * 255
        fg = ((alpha.astype(np.float32) > 100.0).astype(np.uint8)) * 255
        fg = _restrict_to_box(fg, box)
        return SegmentResult(mask=fg, meta={"backend": self.name})


class OnnxMattingSegmenter:
    """
    ONNX 매팅/분할 모델.

    **출력 계약**: 하나의 출력, 형상 `(1,1,H,W)` / `(1,H,W)` / `(H,W)` 의 확률맵.
    다중 채널(클래스별 로짓) 모델은 계약이 달라 형상 검증에서 실패한다.
    """

    name = "onnx"

    def __init__(
        self,
        model_dir: str,
        filename: str = "segmenter.onnx",
        input_size: int = 320,
        threshold: float = 0.5,
        session: object | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.filename = filename
        self.input_size = input_size
        self.threshold = threshold
        self._session = session

    def _sess(self):
        if self._session is None:
            self._session = load_session(self.model_dir, self.filename)
        return self._session

    def segment(self, bgr: np.ndarray, box: Detection | None = None) -> SegmentResult:
        import cv2

        h, w = bgr.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(bgr, self.input_size, pad_value=0)
        tensor = to_nchw_rgb(canvas)

        sess = self._sess()
        raw = np.asarray(sess.run(None, {sess.get_inputs()[0].name: tensor})[0], dtype=np.float32)
        require_shape("segmenter", raw, 2, 3, 4)
        while raw.ndim > 2:
            if raw.shape[0] != 1:
                raise contract_error(
                    "segmenter",
                    f"leading dim {raw.shape[0]} != 1 — 단일 확률맵 계약이 아닙니다"
                    f" (클래스별 로짓 모델은 별도 백엔드 필요)",
                )
            raw = raw[0]

        if raw.max() > 1.0 or raw.min() < 0.0:  # 로짓이면 시그모이드
            raw = 1.0 / (1.0 + np.exp(-raw))

        if raw.shape != (self.input_size, self.input_size):
            raw = cv2.resize(raw, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        nw = max(1, round(w * scale))
        nh = max(1, round(h * scale))
        crop = raw[pad_y : pad_y + nh, pad_x : pad_x + nw]
        prob = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

        fg = ((prob >= self.threshold).astype(np.uint8)) * 255
        fg = _restrict_to_box(fg, box)
        return SegmentResult(
            mask=fg, meta={"backend": self.name, "model": self.filename, "threshold": self.threshold}
        )


def _restrict_to_box(fg: np.ndarray, box: Detection | None) -> np.ndarray:
    """검출 박스 밖은 제거 — 검출기가 있으면 배경 과포함을 크게 줄인다."""
    if box is None:
        return fg
    h, w = fg.shape[:2]
    x0, y0, x1, y1 = box.clipped(h, w).box_xyxy
    keep = np.zeros_like(fg)
    keep[y0 : y1 + 1, x0 : x1 + 1] = 255
    return np.bitwise_and(fg, keep)
