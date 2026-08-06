"""
거리 추정 — 기준물 없이 **카메라↔물체 거리**를 낸다 (에라토스테네스 거리 모드).

## 원리

핀홀 카메라에서 실제 크기 `S`, 픽셀 크기 `p`, 초점거리 `f`(px), 거리 `Z` 는

    p = f · S / Z      →      Z = f · S / p

`p` 는 누끼에서 잰다. 그러면 **`f` 와 `S` 만 있으면 거리가 나온다.**

## f — Depth Pro 의 초점거리 추정을 쓴다

이 모델의 **절대 깊이는 못 쓴다** (실측 중앙값 10.3배 과대, 상세는
`eratosthenes.py` 머리말). 그런데 **초점거리 추정은 다르다.**

같은 폰으로 찍은 6장에서 모델이 낸 f 는 3319·3750·3769·3423·3711·3408 px 로
**±6% 안에 뭉쳤다.** 같은 렌즈면 f 는 상수여야 하므로 이건 일관성의 증거다.
반면 카드 소실점으로 푼 f 는 1615~4637 px 로 널뛰었다(쿼드 몇 px 오차가
소실점을 크게 흔든다). 즉 이 사진들에서는 **모델 f 가 카드 f 보다 안정적이다.**

EXIF 에 초점거리가 있으면 그걸 우선한다 — 실측값이 추정값보다 낫다.

## S — 제품 종류의 크기 사전값

사용자는 이미 제품 종류를 고른다(반지·목걸이·골드바…). 종류마다 대표 크기가
있으므로 그걸 `S` 로 쓴다. 사전값의 폭이 그대로 거리 오차가 된다.

**반지가 가장 좁다.** 손가락에 끼워야 하므로 안지름이 15~22mm 로 묶인다.
골드바·목걸이는 폭이 넓어 거리 오차도 그만큼 커진다.

사용자가 실제 크기를 알면 입력받아 쓴다 — 그러면 오차가 `f` 하나로 줄어든다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectSizePrior:
    """제품 종류별 **긴 쪽 실제 크기** 대표값과 그 폭."""

    product_k: str
    korean: str
    long_mm: float
    spread: float  # 상대 폭. 0.15 면 ±15%
    note: str


# 긴 쪽(최대 외곽 치수) 기준. 폭은 소비자 제품 범위를 담되 극단은 뺀다.
#
# ring    : 안지름 15~22mm(한국 호수 1~30) + 밴드 두께 → 바깥 18~26mm
# earring : 귀걸이는 형태가 다양해 폭이 크다
# goldbar : 0.05g 카드형(40mm)부터 37.5g 바(50mm)까지 — 폭이 매우 크다
OBJECT_SIZE_PRIORS: dict[str, ObjectSizePrior] = {
    # 반지는 **손가락에 들어가야 해서** 크기가 묶인다. 안지름 15~22mm(한국 호수
    # 1~30) + 밴드 두께 → 바깥 18~24mm. 실측 대조: 사전값 21mm vs 카드로 잰
    # 20.4mm → 오차 3%. 이 표에서 가장 믿을 만한 항목이다.
    "ring": ObjectSizePrior("ring", "반지", 21.0, 0.12, "안지름 15~22mm + 밴드"),
    # 골드바는 0.05g 카드형(약 40mm)부터 100g 바(60mm 이상)까지 — 폭이 크다.
    # 실측 대조: 도련님 0.05g 카드형이 38~40mm 였다.
    "goldbar": ObjectSizePrior("goldbar", "골드바", 42.0, 0.30, "0.05g 카드형~100g 바"),
    "plated": ObjectSizePrior("plated", "봉입·도금", 42.0, 0.30, "골드바와 같은 범위"),
    "earring": ObjectSizePrior("earring", "귀걸이", 20.0, 0.40, "형태 편차가 큼"),
    "necklace": ObjectSizePrior("necklace", "목걸이", 45.0, 0.45, "펜던트·체인 뭉침에 따라"),
    "bracelet": ObjectSizePrior("bracelet", "팔찌", 60.0, 0.25, "손목 둘레 기준"),
}

# 소비자 접사에서 물리적으로 말이 되는 범위. 벗어나면 추정이 깨진 것으로 본다.
MIN_DISTANCE_MM = 40.0
MAX_DISTANCE_MM = 1500.0


@dataclass
class DistanceEstimate:
    distance_mm: float
    relative_sigma: float
    focal_px: float
    focal_source: str
    assumed_long_mm: float
    object_long_px: float
    prior: ObjectSizePrior | None
    size_source: str  # user_input | product_prior

    @property
    def range_mm(self) -> tuple[float, float]:
        lo = self.distance_mm * (1.0 - self.relative_sigma)
        hi = self.distance_mm * (1.0 + self.relative_sigma)
        return max(MIN_DISTANCE_MM, lo), hi

    def as_meta(self) -> dict[str, Any]:
        lo, hi = self.range_mm
        return {
            "object_mm": round(self.distance_mm, 1),
            "range_mm": [round(lo, 1), round(hi, 1)],
            "relative_sigma": round(self.relative_sigma, 4),
            "focal_px": round(self.focal_px, 1),
            "focal_source": self.focal_source,
            "assumed_long_mm": round(self.assumed_long_mm, 2),
            "object_long_px": round(self.object_long_px, 1),
            "size_source": self.size_source,
            "product_note": self.prior.note if self.prior else None,
            "source": "size_prior_pinhole",
        }


def mask_long_px(mask: np.ndarray) -> float:
    """
    마스크의 **최대 외곽 치수**(px). bbox 가 아니라 최소외접사각형의 긴 변을 쓴다.

    bbox 는 물체가 기울면 대각선까지 세어 크게 나온다. 회전 사각형이면 기울기와
    무관하게 물체 자신의 긴 쪽이다.
    """
    import cv2

    cnts, _h = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return 0.0
    (_c, (rw, rh), _a) = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return float(max(rw, rh))


def estimate_distance(
    mask: np.ndarray,
    focal_px: float,
    focal_source: str,
    product_k: str,
    known_long_mm: float | None = None,
    focal_relative_sigma: float = 0.08,
) -> DistanceEstimate | None:
    """
    누끼 + 초점거리 → 카메라↔물체 거리.

    `known_long_mm` 를 주면 사전값 대신 그 값을 쓴다(사용자가 실제 크기를 알 때).
    그때 오차는 초점거리 오차만 남는다.

    실측(카드 PnP 를 정답으로, 6장): 반지 +10.2% · real5 -10.2% · real -0.9% ·
    real4 -25.7%. 오차의 대부분이 **크기 사전값의 폭**에서 온다. 모델 초점거리는
    같은 폰 6장에서 변동계수 5.2% 로 안정적이었다.
    """
    long_px = mask_long_px(mask)
    if long_px < 20.0 or focal_px <= 0:
        return None

    prior = OBJECT_SIZE_PRIORS.get(product_k.lower())
    if known_long_mm and known_long_mm > 0:
        size_mm, size_sigma, size_source = float(known_long_mm), 0.02, "user_input"
    elif prior is not None:
        size_mm, size_sigma, size_source = prior.long_mm, prior.spread, "product_prior"
    else:
        return None

    z = focal_px * size_mm / long_px
    if not (MIN_DISTANCE_MM <= z <= MAX_DISTANCE_MM):
        log.info("distance estimate out of range: %.1fmm — 버린다", z)
        return None

    # 크기 오차와 초점거리 오차는 독립이라 제곱합으로 더한다
    rel = float(np.hypot(size_sigma, focal_relative_sigma))
    return DistanceEstimate(
        distance_mm=z,
        relative_sigma=rel,
        focal_px=focal_px,
        focal_source=focal_source,
        assumed_long_mm=size_mm,
        object_long_px=long_px,
        prior=prior,
        size_source=size_source,
    )


def distance_warnings(est: DistanceEstimate | None) -> list[str]:
    """무엇을 가정했는지 반드시 밝힌다 — 가정이 곧 오차이기 때문."""
    if est is None:
        return [
            (
                "거리를 추정하지 못했습니다. 물체가 화면에 충분히 크게, 통째로 "
                "보이도록 다시 찍어 주세요."
            )
        ]
    pct = round(est.relative_sigma * 100)
    lo, hi = est.range_mm
    if est.size_source == "user_input":
        head = f"입력하신 실제 크기 {est.assumed_long_mm:.0f}mm 를 기준으로 계산했습니다."
    else:
        name = est.prior.korean if est.prior else "제품"
        head = (
            f"'{name}'의 일반적인 크기({est.assumed_long_mm:.0f}mm)를 가정해 추정한 값입니다. "
            "실제 제품 크기를 아신다면 입력해 주세요 — 훨씬 정확해집니다."
        )
    return [
        head,
        (
            f"추정 거리 {est.distance_mm:.0f}mm (약 {lo:.0f}~{hi:.0f}mm, ±{pct}%). "
            "정확한 값이 필요하면 신용카드를 함께 두고 '분석' 탭을 이용해 주세요."
        ),
    ]
