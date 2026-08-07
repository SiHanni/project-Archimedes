#!/usr/bin/env python3
"""
ONNX 백엔드 스모크 — 가중치를 꽂았을 때 **계약이 맞는지** 한 번에 확인한다.

모델을 고르는 일(벤치마크)은 사람이 하지만, "꽂았는데 조용히 이상한 값이 나오는"
상황은 여기서 걸러진다. 합성 장면을 넣고 각 백엔드의 출력 형상·범위·소요 시간을 찍는다.

사용:
  export ARCHIMEDES_ONNX_MODEL_DIR=/models
  export ARCHIMEDES_DETECTOR_BACKEND=onnx
  export ARCHIMEDES_SEGMENTATION_BACKEND=onnx
  export ARCHIMEDES_DEPTH_BACKEND=onnx
  python worker/scripts/check_models.py

Docker:
  docker compose exec worker python scripts/check_models.py

종료 코드: 설정된 백엔드 중 하나라도 실패하면 1.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_WORKER = Path(__file__).resolve().parents[1]
if str(_WORKER) not in sys.path:
    sys.path.insert(0, str(_WORKER))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.pipeline.backends import (  # noqa: E402
    get_depth_estimator,
    get_detector,
    get_segmenter,
)
from app.pipeline.exceptions import PipelineError  # noqa: E402


def synthetic_scene(size: int = 1024) -> np.ndarray:
    """카드 + 그 옆 물체 — 실제 촬영 프로토콜과 같은 구도."""
    img = np.full((size, size, 3), 220, np.uint8)
    cw = int(size * 0.42)
    ch = round(cw * 53.98 / 85.60)
    x0, y0 = int(size * 0.10), int(size * 0.34)
    cv2.rectangle(img, (x0, y0), (x0 + cw, y0 + ch), (248, 248, 248), -1)
    cv2.rectangle(img, (x0, y0), (x0 + cw, y0 + ch), (60, 60, 60), 3)
    cv2.circle(img, (int(size * 0.78), int(size * 0.50)), int(size * 0.07), (25, 25, 30), -1)
    return img


def _fmt(ok: bool) -> str:
    return "OK  " if ok else "FAIL"


def main() -> int:
    settings = get_settings()
    img = synthetic_scene()
    failures = 0

    print(f"model_dir = {settings.onnx_model_dir}")
    print(f"backends  = detector:{settings.detector_backend} "
          f"segmenter:{settings.segmentation_backend} depth:{settings.depth_backend}\n")

    # ── 검출 ──
    try:
        det = get_detector(settings)
        t0 = time.perf_counter()
        dets = det.detect(img)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[{_fmt(True)}] detector({det.name})  {len(dets)} boxes  {dt:.0f} ms")
        for d in dets[:3]:
            print(f"         box={d.box_xyxy} score={d.score:.4f} label={d.label}")
    except PipelineError as e:
        failures += 1
        print(f"[{_fmt(False)}] detector: {e.code} — {e}")

    # ── 분할 ──
    mask = None
    try:
        seg = get_segmenter(settings)
        t0 = time.perf_counter()
        out = seg.segment(img, None)
        dt = (time.perf_counter() - t0) * 1000
        mask = out.mask
        frac = float(np.count_nonzero(mask)) / mask.size
        print(f"[{_fmt(True)}] segmenter({seg.name})  shape={mask.shape} "
              f"foreground={frac:.4f}  {dt:.0f} ms")
        if frac <= 0.0 or frac >= 0.95:
            failures += 1
            print("         ⚠️ 전경 비율이 비정상입니다 — 임계값·전처리 확인 필요")
    except PipelineError as e:
        failures += 1
        print(f"[{_fmt(False)}] segmenter: {e.code} — {e}")

    # ── 깊이 ──
    try:
        est = get_depth_estimator(settings)
        t0 = time.perf_counter()
        dm = est.estimate(img)
        dt = (time.perf_counter() - t0) * 1000
        valid = dm.valid_mask()
        finite = dm.depth[valid]
        print(f"[{_fmt(True)}] depth({est.name})  shape={dm.depth.shape} kind={dm.kind.value} "
              f"valid={float(valid.mean()):.3f}  {dt:.0f} ms")
        if finite.size:
            print(f"         range=[{finite.min():.4f}, {finite.max():.4f}] "
                  f"median={np.median(finite):.4f}")
            if float(np.std(finite)) < 1e-6:
                print("         ⚠️ 깊이가 상수입니다 — 스케일 융합에서 s 가 식별되지 않습니다")
        if dm.kind.needs_anchor:
            print("         ℹ️ 스케일 미정 모델 — 카드 앵커 없이는 절대 크기를 못 냅니다")
    except PipelineError as e:
        failures += 1
        print(f"[{_fmt(False)}] depth: {e.code} — {e}")

    print()
    if failures:
        print(f"{failures} backend(s) failed.")
        return 1
    print("all configured backends passed the contract smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
