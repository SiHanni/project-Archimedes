"""
객체 검출 백엔드 (계획서 Step 1 — "박스 영역 추출").

- `stub`: 모델 없이 동작. 전경 휴리스틱으로 최대 연결성분 박스를 낸다.
- `onnx`: YOLO 계열 export 의 표준 출력 `(1, 4+nc, N)` 을 디코딩.
  출력 형상을 검증하므로 계약이 다른 export 는 조용히 통과하지 못한다.
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
from app.pipeline.backends.types import Detection

log = logging.getLogger(__name__)


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        """점수 내림차순 Detection 목록."""
        ...


class StubDetector:
    """
    모델 없이 쓰는 기본 검출기.

    Otsu 전경의 최대 연결성분 박스를 `jewelry` 로 낸다. 카드가 함께 잡히면
    분할 단계의 카드 차집합(§5.2)이 걸러내므로 여기서는 관대하게 둔다.
    실제 정확도는 ONNX 백엔드로 교체해야 나온다.
    """

    name = "stub"

    def __init__(self, min_area_frac: float = 0.0002) -> None:
        self.min_area_frac = min_area_frac

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        import cv2

        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(fg.mean()) > 127:
            fg = 255 - fg
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
        out: list[Detection] = []
        min_area = self.min_area_frac * h * w
        for i in range(1, n):  # 0 = 배경
            x, y, bw, bh, area = stats[i]
            if area < min_area:
                continue
            out.append(
                Detection(
                    box_xyxy=(int(x), int(y), int(x + bw - 1), int(y + bh - 1)),
                    score=float(area) / float(h * w),
                    label="jewelry",
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        return out[:10]


class OnnxYoloDetector:
    """
    YOLO 계열 ONNX (v8/v11 export 관례: 출력 `(1, 4+nc, N)`, 박스는 cxcywh).

    ⚠️ 실제 모델을 붙일 때 반드시 검증할 것: 입력 크기, 정규화, 출력 레이아웃,
    클래스 인덱스 매핑. `class_labels` 로 인덱스→라벨을 주입한다.
    """

    name = "onnx"

    def __init__(
        self,
        model_dir: str,
        filename: str = "detector.onnx",
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_labels: tuple[str, ...] = ("jewelry",),
        channels_first: bool = True,
        session: object | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.filename = filename
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_labels = class_labels
        # True = (1, 4+nc, N) — YOLOv8/11 export 관례. False = (1, N, 4+nc).
        # 차원 크기로 추측하지 않는다: 앵커 수가 적은 export 에서 오탐지하고,
        # 잘못 전치되면 조용히 엉뚱한 박스를 낸다.
        self.channels_first = channels_first
        self._session = session

    def _sess(self):
        if self._session is None:
            self._session = load_session(self.model_dir, self.filename)
        return self._session

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(bgr, self.input_size)
        tensor = to_nchw_rgb(canvas)

        sess = self._sess()
        input_name = sess.get_inputs()[0].name
        raw = sess.run(None, {input_name: tensor})[0]
        out = np.asarray(raw)
        require_shape("detector", out, 3)

        # → (N, 4+nc)
        pred = out[0].transpose(1, 0) if self.channels_first else out[0]
        if pred.shape[1] < 5:
            raise contract_error(
                "detector",
                f"channel dim {pred.shape[1]} < 5 (need cx,cy,w,h + >=1 class). "
                f"channels_first={self.channels_first} 설정을 확인해 주세요",
            )

        boxes_cxcywh = pred[:, :4]
        scores_all = pred[:, 4:]
        cls_idx = scores_all.argmax(axis=1)
        conf = scores_all.max(axis=1)

        keep = conf >= self.conf_threshold
        if not np.any(keep):
            return []
        boxes_cxcywh, conf, cls_idx = boxes_cxcywh[keep], conf[keep], cls_idx[keep]

        cx, cy, bw, bh = boxes_cxcywh.T
        x0 = (cx - bw / 2 - pad_x) / scale
        y0 = (cy - bh / 2 - pad_y) / scale
        x1 = (cx + bw / 2 - pad_x) / scale
        y1 = (cy + bh / 2 - pad_y) / scale
        xyxy = np.stack([x0, y0, x1, y1], axis=1)

        dets: list[Detection] = []
        for i in _nms(xyxy, conf, self.iou_threshold):
            label = (
                self.class_labels[cls_idx[i]]
                if cls_idx[i] < len(self.class_labels)
                else f"class_{cls_idx[i]}"
            )
            x0i, y0i, x1i, y1i = np.rint(xyxy[i]).astype(np.int64).tolist()
            dets.append(
                Detection((x0i, y0i, x1i, y1i), float(conf[i]), label).clipped(h, w)
            )
        return dets


def _nms(xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """단순 greedy NMS (검출 수가 적어 벡터화 이득이 작다)."""
    order = scores.argsort()[::-1]
    areas = (xyxy[:, 2] - xyxy[:, 0]).clip(0) * (xyxy[:, 3] - xyxy[:, 1]).clip(0)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy0 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx1 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy1 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = (xx1 - xx0).clip(0) * (yy1 - yy0).clip(0)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep
