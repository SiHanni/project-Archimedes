"""
검출·분할·깊이 백엔드 레지스트리.

모델 교체가 파이프라인 코드를 건드리지 않도록 세 축을 Protocol 로 분리한다
(`archimedes-v2-single-photo.mdc` §2). 환경변수로 라우팅한다.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.pipeline.backends.depth import (
    ConstantDepthEstimator,
    DepthEstimator,
    OnnxDepthEstimator,
)
from app.pipeline.backends.detector import Detector, OnnxYoloDetector, StubDetector
from app.pipeline.backends.segmenter import (
    HeuristicSegmenter,
    OnnxMattingSegmenter,
    RembgSegmenter,
    Segmenter,
)
from app.pipeline.backends.types import DepthKind, DepthMap, Detection, SegmentResult
from app.pipeline.exceptions import PipelineError

log = logging.getLogger(__name__)

__all__ = [
    "ConstantDepthEstimator",
    "DepthEstimator",
    "DepthKind",
    "DepthMap",
    "Detection",
    "Detector",
    "HeuristicSegmenter",
    "OnnxDepthEstimator",
    "OnnxMattingSegmenter",
    "OnnxYoloDetector",
    "RembgSegmenter",
    "SegmentResult",
    "Segmenter",
    "StubDetector",
    "get_depth_estimator",
    "get_detector",
    "get_segmenter",
]


def _unknown(kind: str, value: str, allowed: tuple[str, ...]) -> PipelineError:
    return PipelineError(
        "ERR_MODEL_UNAVAILABLE",
        f"Unknown {kind} backend {value!r}. allowed={list(allowed)}",
    )


def get_detector(settings: Settings) -> Detector:
    b = (settings.detector_backend or "stub").strip().lower()
    if b == "stub":
        return StubDetector()
    if b == "onnx":
        return OnnxYoloDetector(
            model_dir=settings.onnx_model_dir, filename=settings.detector_model_file
        )
    raise _unknown("detector", b, ("stub", "onnx"))


def get_segmenter(settings: Settings) -> Segmenter:
    b = (settings.segmentation_backend or "heuristic").strip().lower()
    if b == "heuristic":
        return HeuristicSegmenter()
    if b == "rembg":
        return RembgSegmenter()
    if b == "onnx":
        return OnnxMattingSegmenter(
            model_dir=settings.onnx_model_dir, filename=settings.segmenter_model_file
        )
    raise _unknown("segmenter", b, ("heuristic", "rembg", "onnx"))


def get_depth_estimator(settings: Settings) -> DepthEstimator:
    b = (settings.depth_backend or "stub").strip().lower()
    if b == "stub":
        return ConstantDepthEstimator()
    if b == "onnx":
        return OnnxDepthEstimator(
            model_dir=settings.onnx_model_dir,
            filename=settings.depth_model_file,
            input_size=settings.depth_input_size,
            kind=DepthKind(settings.depth_output_kind),
            inverse=settings.depth_output_inverse,
            output_scale_to_mm=settings.depth_output_scale_to_mm,
        )
    raise _unknown("depth", b, ("stub", "onnx"))
