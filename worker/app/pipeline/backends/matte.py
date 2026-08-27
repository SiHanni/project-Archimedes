"""
학습 기반 누끼 (BiRefNet ONNX) — 계획서 Step 1 의 세미-오토 라벨링 본선.

## 왜 색 임계값을 버렸는가

기존 경로는 명도(Otsu)·채도로 전경을 만들고 GrabCut 으로 다듬었다. 도련님 실사진
10장으로 실측한 결과가 이렇다.

    T192 목걸이       펜던트 **고리만** 잡음 (0.66%)
    T330 저울 위 목걸이  저울의 **초록 LCD** 를 잡음
    T332 케이스 속 반지   케이스 **모서리·그림자** 를 잡음
    T341 귀걸이 2개     배경까지 사각형으로 뭉갬
    T384 책상 위 반지    아예 실패 (ERR_NO_JEWEL)
    T390 상자 속 금괴    위쪽 조각만 (0.62%)

색 분포로는 (a) 체인처럼 **가는 선**, (b) 상자·저울 위처럼 **배경이 물체보다
눈에 띄는** 구도, (c) 반지 **구멍** 을 원리적으로 못 가른다. 임계값을 어떻게
흔들어도 이 셋 중 하나는 깨진다.

## 왜 BiRefNet 인가

두 후보를 같은 10장에 돌려 비교했다.

    RMBG-1.4 (IS-Net)  10장 중 5장 실패 — 저울 전체(66.8%)·화면 전체(99.99%)
    BiRefNet fp16      10장 전부 물체만, **반지 구멍·체인 고리까지 파냄**

BiRefNet 은 dichotomous image segmentation(고해상도 이분할) 용으로 학습돼
가는 구조와 구멍을 그대로 살린다. MIT 라이선스라 성과물에 넣기도 안전하다.
(RMBG-1.4 는 BRIA RAIL — 비상업 제한이 있어 어차피 못 쓴다)

CPU 로 한 장 7~9초. 누끼 탭은 Depth Pro 를 안 타므로 이 정도는 감당한다.

⚠️ **fp16(`model_fp16.onnx`, 467MB) 을 쓰고, onnxruntime 을 `<1.29` 로 고정한다.**
1.29 는 CPU 에서 `com.microsoft.Gelu` 의 float16 커널이 없어 세션 생성부터
실패한다. 그렇다고 fp32(928MB)로 가면 도련님 Docker VM(7.65GiB)에서 워커가
조용히 SIGKILL 당한다 — `OOMKilled=false` · `ExitCode=0` 으로 찍혀 정상 종료처럼
보이고, 잡은 영원히 processing 에 남는다(실측 RestartCount=3, 잡 192초 미완).
그래서 **런타임을 내리고 가중치를 가볍게** 가는 조합이 유일한 답이다.

## 계약

    input_image  (1,3,1024,1024) float32, ImageNet 정규화
    output_image (1,1,1024,1024) float32 — **로짓이다. sigmoid 를 걸어야 한다**

⚠️ 출력은 확률이 아니라 로짓이다(실측 범위 -82 ~ +21). 그대로 0.5 로 자르면
얼추 맞아 보여서 놓치기 쉬운데, 경계가 뭉개진다. sigmoid 를 먼저 건다.

⚠️ min-max 정규화를 걸지 말 것. 물체가 없는 사진에서 잡음이 1.0 까지 늘어나
화면 전체가 물체가 된다(실측: 이 버그로 10장 중 6장이 99.9% 가 나왔다).
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
# 알파 임계. 0.5 는 sigmoid 의 중립점이라 모델이 학습한 경계와 일치한다.
_ALPHA_THRESHOLD = 0.5
# 2차 크롭에 남기는 여백 (긴 변 대비). 경계가 프레임에 붙으면 모델이 잘린 것으로 본다.
_CROP_PAD_RATIO = 0.06
# 2차 결과가 1차 대비 이 범위를 벗어나면 버린다.
# 실측 10장의 실제 비율은 0.28~1.11 이었다 — 0.28(저울 벗김)은 살리고,
# 물체가 통째로 사라지는(0에 가까운) 경우만 걸러 낸다.
_MIN_REFINE_RATIO = 0.2
_MAX_REFINE_RATIO = 1.3


class BiRefNetMatter:
    """사진 한 장 → 알파 맷. 임계값이 없어 조명·배경에 흔들리지 않는다."""

    name = "birefnet"

    def __init__(self, model_dir: str, model_file: str = "model_fp16.onnx") -> None:
        self.model_dir = model_dir
        self.model_file = model_file
        self._sess: Any = None

    def _session(self) -> Any:
        if self._sess is None:
            from app.pipeline.backends.onnx_session import load_session

            self._sess = load_session(self.model_dir, self.model_file)
        return self._sess

    def alpha(self, bgr: np.ndarray) -> np.ndarray:
        """원본 해상도 알파 [0,1]."""
        h, w = bgr.shape[:2]
        sess = self._session()

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = (x.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        logit = np.asarray(out, dtype=np.float32).reshape(out.shape[-2], out.shape[-1])
        # ⚠️ 로짓 → 확률. min-max 정규화가 아니다 (머리말 참조)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

    def mask(self, bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """(이진 마스크 0/255, meta). 구멍은 **그대로 남긴다.**"""
        a = self.alpha(bgr)
        binary = ((a > _ALPHA_THRESHOLD).astype(np.uint8)) * 255
        frac = float(cv2.countNonZero(binary)) / float(binary.size)
        return binary, {
            "backend": self.name,
            "alpha_max": round(float(a.max()), 3),
            "area_frac": round(frac, 6),
        }

    def mask_refined(self, bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """
        1차 마스크의 bbox 로 잘라 **한 번 더 돌린다.** 용기를 벗겨내는 단계다.

        귀금속은 저울 위·케이스 안에 놓고 찍는 일이 많은데, 그 구도에서는 **용기가
        주인공**이라 모델이 용기를 잡는다(실측: T330 저울 몸통까지 19.75%,
        T390 선물상자까지 26.73%). 1차 결과로 크롭하면 용기가 화면을 꽉 채워
        배경이 되고, 그 안의 귀금속이 주인공으로 올라온다.

        실측 10장 — T330 19.75%→5.59%(저울 몸통 빠짐), T390 26.73%→16.11%(상자
        빠짐), T152 3.52%→2.36%·T332 2.80%→2.14%(반사光이 빠져 더 조여짐).
        나머지 6장은 0.99~1.11 배로 사실상 그대로였다. 즉 **손해 보는 사진이 없다.**

        다만 크롭이 물체를 꽉 채우면 모델이 되레 헤맬 수 있으므로, 2차 결과가
        1차 대비 아래 범위를 벗어나면 1차를 쓴다.
        """
        first, meta = self.mask(bgr)
        h, w = bgr.shape[:2]
        ys, xs = np.where(first > 0)
        if not ys.size:
            return first, meta

        pad = int(_CROP_PAD_RATIO * max(h, w))
        y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
        x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
        if (y1 - y0) < 32 or (x1 - x0) < 32:
            return first, meta

        try:
            crop_mask, _ = self.mask(bgr[y0:y1, x0:x1])
        except Exception as e:  # noqa: BLE001 — 2차는 어디까지나 개선이다
            log.warning("matte second pass failed: %s", e)
            return first, meta

        second = np.zeros((h, w), np.uint8)
        second[y0:y1, x0:x1] = crop_mask

        a1 = int(cv2.countNonZero(first))
        a2 = int(cv2.countNonZero(second))
        ratio = a2 / float(a1) if a1 else 0.0
        meta["refine_ratio"] = round(ratio, 3)
        if not (_MIN_REFINE_RATIO <= ratio <= _MAX_REFINE_RATIO):
            meta["refine"] = "rejected"
            return first, meta

        meta["refine"] = "crop_rerun"
        meta["area_frac"] = round(a2 / float(h * w), 6)
        return second, meta
