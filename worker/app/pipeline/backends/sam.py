"""
프롬프트 기반 분할 (SlimSAM ONNX) — 계획서 Step 1 의 학습 기반 분할.

## 왜 필요한가

외형 폴백(Otsu·채도 + GrabCut)은 **임계값이 조명에 흔들린다.** 실측: 같은 반지를
찍은 두 장에서 씨앗 면적이 1.379% vs 2.139%(1.55배), GrabCut 이 그림자를 먹어
최종 2.32% vs 4.34%(1.87배). 색 분포로 다듬어도 1.78배까지밖에 못 줄였다.
임계값이 없는 방법이 필요하다.

SAM 계열은 "여기 있는 물체"라는 **프롬프트**만 받고 경계는 학습된 표현으로
찾는다. 임계값이 없으니 조명이 바뀌어도 같은 물체면 같은 마스크가 나온다.

## 모델

`Xenova/slimsam-77-uniform` — SAM 을 77% 가지치기한 경량본. 인코더 23MB +
디코더 17MB 로 40MB. 접사 한 장에 CPU 로도 부담이 없다.

## 계약

    vision_encoder            pixel_values(1,3,1024,1024)
                              → image_embeddings, image_positional_embeddings
    prompt_encoder_mask_decoder  위 둘 + input_points(1,1,N,2, float32)
                                       + input_labels(1,1,N, **int64**)
                              → iou_scores(1,1,3), pred_masks(1,1,3,256,256)

⚠️ 좌표는 **1024 로 리사이즈된 좌표계**다. 원본 픽셀을 그대로 넣으면 프롬프트가
엉뚱한 데를 가리킨다. 그리고 리사이즈는 **가로세로 비를 유지하지 않는다**
(`longest_edge` 지만 export 는 정사각 고정) — x·y 를 각각 배율해야 한다.

⚠️ 마스크 3개 중 `iou_scores` 최대를 고른다. SAM 은 "부분/전체/더 큰 전체"를
동시에 내는데, 점 하나만 주면 부분이 나올 수 있어 점수로 골라야 한다.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

_INPUT_SIZE = 1024
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# 디코더 로짓 임계. SAM 관례상 0 이 경계다.
_MASK_LOGIT_THRESHOLD = 0.0


class SlimSamSegmenter:
    """점 프롬프트로 물체를 분할한다. 임계값이 없어 조명에 흔들리지 않는다."""

    name = "slimsam"

    def __init__(
        self,
        model_dir: str,
        encoder_file: str = "vision_encoder.onnx",
        decoder_file: str = "prompt_encoder_mask_decoder.onnx",
    ) -> None:
        self.model_dir = model_dir
        self.encoder_file = encoder_file
        self.decoder_file = decoder_file
        self._enc: Any = None
        self._dec: Any = None

    def _sessions(self) -> tuple[Any, Any]:
        if self._enc is None or self._dec is None:
            from app.pipeline.backends.onnx_session import load_session

            self._enc = load_session(self.model_dir, self.encoder_file)
            self._dec = load_session(self.model_dir, self.decoder_file)
        return self._enc, self._dec

    def segment_at(
        self, bgr: np.ndarray, points_xy: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        (원본 해상도 마스크, iou 점수).

        `points_xy` 는 **원본 픽셀 좌표** (N,2), `labels` 는 1=전경 / 0=배경.
        """
        h, w = bgr.shape[:2]
        enc, dec = self._sessions()

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...].astype(np.float32)

        emb, pos = enc.run(None, {enc.get_inputs()[0].name: tensor})

        # 원본 → 1024 좌표. x·y 배율이 다르다(정사각 리사이즈).
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2).copy()
        pts[:, 0] *= _INPUT_SIZE / float(w)
        pts[:, 1] *= _INPUT_SIZE / float(h)

        feeds = {
            "input_points": pts[None, None, ...].astype(np.float32),
            # ⚠️ 라벨은 **int64** 다. float32 로 넣으면 ONNX Runtime 이 거부한다.
            "input_labels": np.asarray(labels, dtype=np.int64).reshape(1, 1, -1),
            "image_embeddings": emb,
            "image_positional_embeddings": pos,
        }
        iou, masks = dec.run(None, feeds)

        iou = np.asarray(iou).reshape(-1)
        masks = np.asarray(masks)[0, 0]  # (3, 256, 256)
        best = int(np.argmax(iou))
        logit = masks[best]

        # 256 → 원본. 로짓을 키운 뒤 임계하면 경계가 부드럽다.
        logit = cv2.resize(logit.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = ((logit > _MASK_LOGIT_THRESHOLD).astype(np.uint8)) * 255
        return mask, float(iou[best])


def prompt_from_mask(seed: np.ndarray, n_points: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    씨앗 마스크 → SAM 점 프롬프트.

    **무게중심을 그대로 쓰면 안 된다.** 반지처럼 가운데가 빈 물체는 무게중심이
    구멍 한복판(=바닥)이라 SAM 이 바닥을 물체로 잡는다. 마스크 안에서 경계로부터
    가장 먼 점(거리변환 최대)을 쓰면 반드시 **물체 위**에 찍힌다.

    점을 여러 개 주면 물체 전체를 가리켜 부분만 잡히는 일이 준다.
    """
    dist = cv2.distanceTransform((seed > 0).astype(np.uint8), cv2.DIST_L2, 5)
    pts: list[tuple[float, float]] = []
    work = dist.copy()
    for _ in range(max(1, n_points)):
        _mn, mx, _mnl, mxl = cv2.minMaxLoc(work)
        if mx <= 0:
            break
        pts.append((float(mxl[0]), float(mxl[1])))
        # 뽑은 점 주변은 지워 다음 점이 다른 곳에서 나오게 한다
        cv2.circle(work, mxl, max(3, int(mx * 3)), 0, -1)
    if not pts:
        ys, xs = np.where(seed > 0)
        pts = [(float(xs.mean()), float(ys.mean()))]
    return np.array(pts, dtype=np.float32), np.ones(len(pts), dtype=np.int64)


# SAM 결과가 씨앗 대비 이 범위를 벗어나면 프롬프트를 잘못 짚은 것으로 본다.
# 실측: 씨앗이 2mm² 짜리 잡음이었을 때 SAM 이 12,984mm²(책상 전체)를 냈다.
# SAM 은 프롬프트가 가리키는 곳을 믿으므로, **씨앗이 나쁘면 결과도 나쁘다.**
_MIN_SAM_GROWTH = 0.4
_MAX_SAM_GROWTH = 2.5


def refine_with_sam(
    segmenter: SlimSamSegmenter,
    bgr: np.ndarray,
    seed: np.ndarray,
    *,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    씨앗을 프롬프트로 SAM 분할. 씨앗과 규모가 크게 다르면 **씨앗을 그대로 돌려준다.**

    Returns (mask, meta).
    """
    seed_area = int(cv2.countNonZero(seed))
    meta: dict[str, Any] = {"sam": "used"}
    if seed_area < 64:
        return seed, {"sam": "skipped_small_seed"}

    try:
        pts, lbs = prompt_from_mask(seed)
        mask, iou = segmenter.segment_at(bgr, pts, lbs)
    except Exception as e:  # noqa: BLE001 — 분할 실패가 분석 실패가 되면 안 된다
        log.warning("sam failed: %s", e)
        return seed, {"sam": "failed"}

    if exclude is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude))

    # 씨앗과 이어진 성분만 — SAM 이 여러 덩어리를 낼 수 있다
    n, labels, _st, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return seed, {"sam": "empty"}
    overlaps = [int(np.count_nonzero((labels == i) & (seed > 0))) for i in range(1, n)]
    if max(overlaps) == 0:
        return seed, {"sam": "lost_seed"}
    mask = ((labels == 1 + int(np.argmax(overlaps))).astype(np.uint8)) * 255

    growth = int(cv2.countNonZero(mask)) / float(seed_area)
    meta.update({"sam_iou": round(iou, 3), "sam_growth": round(growth, 3)})
    if not (_MIN_SAM_GROWTH <= growth <= _MAX_SAM_GROWTH):
        meta["sam"] = "rejected_growth"
        return seed, meta
    return mask, meta
