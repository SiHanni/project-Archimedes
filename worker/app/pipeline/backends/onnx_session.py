"""
ONNX Runtime 세션 공용 로더 (CPU 기준).

모델 가중치는 **이미지에 굽지 않고** `ARCHIMEDES_ONNX_MODEL_DIR` 볼륨에서 주입한다
(`archimedes-v2-single-photo.mdc` §2). 파일이 없으면 조용히 넘어가지 않고
`ERR_MODEL_UNAVAILABLE` 로 즉시 실패한다 — 잘못된 값을 내는 것보다 낫다.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from app.pipeline.exceptions import PipelineError

log = logging.getLogger(__name__)

_SESSION_CACHE: dict[str, Any] = {}


def resolve_model_path(model_dir: str, filename: str) -> str:
    path = os.path.join(model_dir, filename)
    if not os.path.isfile(path):
        raise PipelineError(
            "ERR_MODEL_UNAVAILABLE",
            f"ONNX model not found: {path}. "
            f"ARCHIMEDES_ONNX_MODEL_DIR 볼륨에 가중치를 넣거나 해당 백엔드를 "
            f"stub/heuristic 으로 되돌려 주세요.",
        )
    return path


def load_session(model_dir: str, filename: str, *, cache: bool = True) -> Any:
    key = os.path.join(model_dir, filename)
    if cache and key in _SESSION_CACHE:
        return _SESSION_CACHE[key]

    path = resolve_model_path(model_dir, filename)
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise PipelineError(
            "ERR_MODEL_UNAVAILABLE",
            'onnxruntime 미설치. `pip install -e ".[onnx]"` 후 다시 시도해 주세요.',
        ) from e

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 워커는 job 병렬로 스케일아웃하므로 세션 내부 스레드는 억제한다
    opts.intra_op_num_threads = int(os.environ.get("ARCHIMEDES_ONNX_THREADS", "2"))
    # CPU 메모리 아레나는 해제한 블록을 붙들고 있어 **최대 상주가 크게 뛴다.**
    # 컨테이너 메모리가 빠듯한 환경(도련님 Docker VM 7.7GiB)에서는 큰 모델이
    # 조용히 SIGKILL 당한다 — 예외도 로그도 안 남아 원인 찾기가 어렵다.
    if os.environ.get("ARCHIMEDES_ONNX_DISABLE_ARENA", "1") == "1":
        opts.enable_cpu_mem_arena = False
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    log.info("loaded ONNX model %s inputs=%s", path, [i.name for i in sess.get_inputs()])
    if cache:
        _SESSION_CACHE[key] = sess
    return sess


def letterbox(
    bgr: np.ndarray, size: int, pad_value: int = 114
) -> tuple[np.ndarray, float, int, int]:
    """
    종횡비 유지 리사이즈 + 패딩.

    Returns (canvas, scale, pad_x, pad_y) — 역변환은 `(px - pad_x) / scale`.
    """
    import cv2

    h, w = bgr.shape[:2]
    scale = min(size / float(w), size / float(h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=bgr.dtype)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def to_nchw_rgb(
    canvas: np.ndarray,
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """BGR uint8 HWC → RGB float32 NCHW, [0,1] 정규화 후 mean/std 적용."""
    import cv2

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])


def contract_error(name: str, detail: str) -> PipelineError:
    return PipelineError(
        "ERR_MODEL_UNAVAILABLE",
        f"ONNX output contract violated for {name}: {detail}. "
        f"이 백엔드의 출력 계약과 다른 모델입니다 — export 형식을 확인해 주세요.",
    )


def require_shape(name: str, arr: np.ndarray, *allowed_ndim: int) -> None:
    """
    모델 출력 형상을 **명시적으로 검증**한다.

    ONNX export 는 모델마다 출력 레이아웃이 달라, 검증 없이 디코딩하면 조용히
    쓰레기 값을 낸다. 형상이 계약과 다르면 즉시 실패시킨다.
    """
    if arr.ndim not in allowed_ndim:
        raise contract_error(
            name, f"got shape {arr.shape} (expected ndim in {allowed_ndim})"
        )
