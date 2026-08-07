"""검출·분할·깊이 백엔드 계약 테스트.

ONNX 경로는 가짜 세션을 주입해 **onnxruntime·가중치 없이도** 디코딩과
형상 검증을 검사한다. 핵심 관심사: 계약이 다른 모델이 조용히 통과하지 않을 것.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline.backends import (
    ConstantDepthEstimator,
    DepthEstimator,
    Detector,
    HeuristicSegmenter,
    OnnxDepthEstimator,
    OnnxMattingSegmenter,
    OnnxYoloDetector,
    Segmenter,
    StubDetector,
    get_depth_estimator,
    get_detector,
    get_segmenter,
)
from app.pipeline.backends.types import DepthKind, Detection
from app.pipeline.exceptions import PipelineError


class FakeSession:
    """onnxruntime InferenceSession 최소 흉내."""

    def __init__(self, output: np.ndarray, input_name: str = "images") -> None:
        self._output = output
        self._input_name = input_name

    def get_inputs(self):
        return [type("I", (), {"name": self._input_name})()]

    def run(self, _outputs, _feed):
        return [self._output]


def _scene(h: int = 240, w: int = 320) -> np.ndarray:
    img = np.full((h, w, 3), 220, np.uint8)
    cv2.circle(img, (w // 2, h // 2), 30, (20, 20, 25), -1)
    return img


# ───────────────────────── Protocol 준수 ─────────────────────────


def test_backends_satisfy_protocols() -> None:
    assert isinstance(StubDetector(), Detector)
    assert isinstance(HeuristicSegmenter(), Segmenter)
    assert isinstance(ConstantDepthEstimator(), DepthEstimator)
    assert isinstance(OnnxYoloDetector("/models", session=FakeSession(np.zeros((1, 5, 1)))), Detector)


def test_registry_routes_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHIMEDES_DETECTOR_BACKEND", "stub")
    monkeypatch.setenv("ARCHIMEDES_SEGMENTATION_BACKEND", "heuristic")
    monkeypatch.setenv("ARCHIMEDES_DEPTH_BACKEND", "stub")
    s = Settings()
    assert get_detector(s).name == "stub"
    assert get_segmenter(s).name == "heuristic"
    assert get_depth_estimator(s).name == "stub"


def test_registry_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHIMEDES_DEPTH_BACKEND", "magic")
    with pytest.raises(PipelineError) as ei:
        get_depth_estimator(Settings())
    assert ei.value.code == "ERR_MODEL_UNAVAILABLE"


# ───────────────────────── 스텁 동작 ─────────────────────────


def test_stub_detector_finds_object() -> None:
    dets = StubDetector().detect(_scene())
    assert dets, "어두운 원을 찾아야 한다"
    x0, y0, x1, y1 = dets[0].box_xyxy
    assert x0 < 160 < x1 and y0 < 120 < y1


def test_constant_depth_is_affine_invariant() -> None:
    """스텁은 스케일 미정이라고 **선언**해야 앵커 없이 쓰이지 않는다."""
    dm = ConstantDepthEstimator().estimate(_scene())
    assert dm.kind is DepthKind.AFFINE_INVARIANT
    assert dm.kind.needs_anchor
    assert dm.depth.shape == (240, 320)
    assert dm.valid_mask().all()


def test_metric_depth_does_not_need_anchor() -> None:
    assert not DepthKind.METRIC.needs_anchor


def test_segmenter_box_restriction() -> None:
    """검출 박스 밖 전경은 잘려야 한다 — 배경 과포함 완화."""
    img = _scene()
    full = HeuristicSegmenter().segment(img).mask
    boxed = HeuristicSegmenter().segment(img, Detection((0, 0, 40, 40), 1.0, "jewelry")).mask
    assert int((boxed > 0).sum()) < int((full > 0).sum())
    assert int((boxed[60:, 60:] > 0).sum()) == 0


# ───────────────────── ONNX 디코딩·계약 검증 ─────────────────────


def test_onnx_depth_decodes_and_resizes() -> None:
    out = np.linspace(1.0, 2.0, 64 * 64, dtype=np.float32).reshape(1, 1, 64, 64)
    est = OnnxDepthEstimator("/models", input_size=64, session=FakeSession(out))
    dm = est.estimate(_scene(120, 160))
    assert dm.depth.shape == (120, 160)
    assert np.isfinite(dm.depth).all()


def test_onnx_depth_inverse_flag_flips_ordering() -> None:
    """역깊이 모델을 그냥 쓰면 물체가 뒤집혀 복원된다 — 플래그가 실제로 반전하는지."""
    out = np.array([[[[1.0, 4.0], [2.0, 8.0]]]], dtype=np.float32)
    direct = OnnxDepthEstimator("/models", input_size=2, session=FakeSession(out)).estimate(
        _scene(2, 2)
    )
    inverse = OnnxDepthEstimator(
        "/models", input_size=2, inverse=True, session=FakeSession(out)
    ).estimate(_scene(2, 2))
    assert direct.depth.argmax() == inverse.depth.argmin()


def test_onnx_depth_rejects_wrong_output_rank() -> None:
    bad = np.zeros((1, 3, 8, 8, 2), dtype=np.float32)  # ndim=5 → 계약 위반
    est = OnnxDepthEstimator("/models", input_size=8, session=FakeSession(bad))
    with pytest.raises(PipelineError) as ei:
        est.estimate(_scene(16, 16))
    assert ei.value.code == "ERR_MODEL_UNAVAILABLE"


def test_onnx_segmenter_thresholds_probability() -> None:
    prob = np.zeros((1, 1, 32, 32), dtype=np.float32)
    prob[0, 0, :16, :] = 1.0
    seg = OnnxMattingSegmenter("/models", input_size=32, session=FakeSession(prob))
    mask = seg.segment(_scene(32, 32)).mask
    assert mask.shape == (32, 32)
    assert int((mask > 0).sum()) > 0


def test_onnx_detector_decodes_standard_layout() -> None:
    """YOLO 관례 (1, 4+nc, N), cxcywh. 640 정사각 입력이라 letterbox 가 항등."""
    pred = np.zeros((1, 5, 2), dtype=np.float32)
    pred[0, :, 0] = [320.0, 320.0, 64.0, 64.0, 0.9]  # 중앙 박스, conf 0.9
    pred[0, :, 1] = [10.0, 10.0, 4.0, 4.0, 0.01]  # conf 미달 → 탈락
    det = OnnxYoloDetector("/models", input_size=640, session=FakeSession(pred))
    got = det.detect(_scene(640, 640))
    assert len(got) == 1
    assert got[0].label == "jewelry"
    assert got[0].score == pytest.approx(0.9, abs=1e-5)
    assert got[0].box_xyxy == (288, 288, 352, 352)


def test_onnx_detector_layout_is_explicit_not_guessed() -> None:
    """
    레이아웃을 차원 크기로 추측하면 앵커 수가 적은 export 에서 오전치되어
    조용히 엉뚱한 박스를 낸다 → `channels_first` 로 명시받는다.
    """
    pred_nc_last = np.zeros((1, 2, 5), dtype=np.float32)
    pred_nc_last[0, 0] = [320.0, 320.0, 64.0, 64.0, 0.9]
    det = OnnxYoloDetector(
        "/models", input_size=640, channels_first=False, session=FakeSession(pred_nc_last)
    )
    assert det.detect(_scene(640, 640))[0].box_xyxy == (288, 288, 352, 352)

    wrong = OnnxYoloDetector(
        "/models", input_size=640, channels_first=True, session=FakeSession(pred_nc_last)
    )
    with pytest.raises(PipelineError) as ei:
        wrong.detect(_scene(640, 640))
    assert ei.value.code == "ERR_MODEL_UNAVAILABLE"


def test_onnx_detector_rejects_wrong_output_rank() -> None:
    det = OnnxYoloDetector("/models", session=FakeSession(np.zeros((5, 2), dtype=np.float32)))
    with pytest.raises(PipelineError) as ei:
        det.detect(_scene())
    assert ei.value.code == "ERR_MODEL_UNAVAILABLE"


def test_missing_model_file_fails_loudly() -> None:
    """가중치가 없으면 조용히 넘어가지 않고 즉시 실패해야 한다."""
    est = OnnxDepthEstimator("/definitely/not/here", filename="depth.onnx")
    with pytest.raises(PipelineError) as ei:
        est.estimate(_scene())
    assert ei.value.code == "ERR_MODEL_UNAVAILABLE"
    assert "not found" in str(ei.value)
