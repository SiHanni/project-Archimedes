#!/usr/bin/env python
"""
metric depth 모델의 **절대 거리 정확도**를 카드 실측으로 채점한다.

## 왜 이 검증이 먼저인가

카드 없이 거리를 재려면 모델이 미터 단위를 스스로 맞혀야 한다. 맞히는지 여부는
**측정해야 알 수 있다.** Depth Pro 는 실내 수 미터 장면으로 학습됐고 우리가 쓰는
구도는 10~30cm 접사라 학습 분포 밖이다.

다행히 정답이 있다. 아르키메데스의 카드 앵커 경로가 **카드 평면까지의 절대 거리**를
PnP 로 이미 낸다(ID-1 85.60×53.98mm 는 국제 규격이라 실측값이다). 같은 사진에서
Depth Pro 가 카드 영역을 얼마로 보는지 비교하면 오차가 그대로 나온다.

## 쓰는 법

    python scripts/eval_metric_distance.py /tmp/archi_real/*.jpg

카드가 검출된 사진만 채점한다(정답이 없으면 채점할 수 없으므로).
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings
from app.pipeline.camera import intrinsics_from_card
from app.pipeline.card import try_compute_card_geometry
from app.pipeline.ingest import bytes_to_bgr, collect_exif
from app.pipeline.metric_depth import DepthProEstimator
from app.pipeline.scale_fusion import solve_card_plane


def card_truth_mm(card, K) -> float | None:
    """
    카드 PnP 로 얻은 **카드 중심까지의 거리**(mm). 이게 정답이다.

    `solve_card_plane` 은 카메라 좌표계 평면 `(n, d)` 를 준다(`n·X = d`).
    중심 픽셀의 광선 `r = ((u-cx)/fx, (v-cy)/fy, 1)` 위의 점은 `Z·r` 이므로
    `Z = d / (n·r)`.
    """
    try:
        normal, d_mm = solve_card_plane(card, K)
    except Exception:  # noqa: BLE001 — 채점 스크립트는 한 장 실패로 멈추지 않는다
        return None
    cx = float(card.quad_px[:, 0].mean())
    cy = float(card.quad_px[:, 1].mean())
    ray = np.array([(cx - K.cx) / K.fx, (cy - K.cy) / K.fy, 1.0])
    denom = float(np.dot(normal, ray))
    if abs(denom) < 1e-9:
        return None
    return float(d_mm / denom)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model-dir", default=os.environ.get("ARCHIMEDES_DEPTHPRO_DIR", "/models/depthpro"))
    ap.add_argument("--model-file", default=os.environ.get("ARCHIMEDES_DEPTHPRO_FILE", "model_quantized.onnx"))
    args = ap.parse_args()

    est = DepthProEstimator(args.model_dir, args.model_file)
    settings = Settings()

    rows = []
    for path in args.images:
        name = os.path.basename(path)
        with open(path, "rb") as f:
            raw = f.read()
        exif = collect_exif(raw)
        bgr = bytes_to_bgr(raw, exif.get("orientation"))
        h, w = bgr.shape[:2]

        card = try_compute_card_geometry(bgr, "front", settings, exif)
        if card is None:
            print(f"{name:16s} 카드 미검출 — 채점 불가")
            continue
        K = intrinsics_from_card(exif, card.quad_px, w, h)
        truth = card_truth_mm(card, K)
        if truth is None:
            print(f"{name:16s} 카드 PnP 실패 — 채점 불가")
            continue

        md = est.estimate(bgr)
        card_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(card_mask, [np.asarray(card.quad_px, dtype=np.int32)], 255)
        # 테두리 오차를 피해 안쪽만 본다
        k = max(3, round(min(h, w) * 0.01) | 1)
        card_mask = cv2.erode(card_mask, np.ones((k, k), np.uint8))
        sel = (card_mask > 0) & np.isfinite(md.depth_m)
        if int(np.count_nonzero(sel)) < 64:
            print(f"{name:16s} 깊이 표본 부족")
            continue
        pred = float(np.median(md.depth_m[sel])) * 1000.0

        err = (pred - truth) / truth * 100.0
        rows.append((name, truth, pred, err, md.focal_px, K.fx, md.fov_deg))
        print(
            f"{name:16s} 정답 {truth:7.1f}mm  예측 {pred:8.1f}mm  오차 {err:+7.1f}%   "
            f"f_모델 {md.focal_px:7.0f}px  f_카드 {K.fx:7.0f}px  화각 {md.fov_deg:.1f}°"
        )

    if not rows:
        print("\n채점 가능한 사진이 없습니다.")
        return 1

    errs = np.array([r[3] for r in rows])
    ratios = np.array([r[2] / r[1] for r in rows])
    print(f"\n표본 {len(rows)}장")
    print(f"  절대오차 중앙값 {np.median(np.abs(errs)):.1f}%   최대 {np.max(np.abs(errs)):.1f}%")
    print(f"  예측/정답 비율  중앙값 {np.median(ratios):.3f}   표준편차 {np.std(ratios):.3f}")
    # 비율이 일정하면 상수배만 보정해도 쓸 만하다는 뜻이다
    corrected = ratios / np.median(ratios)
    print(f"  ⤷ 중앙값으로 보정하면 잔차 {np.max(np.abs(corrected - 1)) * 100:.1f}% 이내")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
