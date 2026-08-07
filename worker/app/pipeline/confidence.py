from __future__ import annotations

import math
from dataclasses import dataclass

from app.constants import PRIOR_MASS_G


@dataclass
class ConfidenceState:
    multires_penalty: bool = False
    scale_tight: bool = True
    quality_ok: bool = True
    # §3 Precision: 다뷰에서 호모그래피 분해 후보 인정 시 한 단계 보정(과장 금지)
    precision_boost: bool = False
    # 부피 모델이 상한 근사(슬랩 AABB)이거나 스케일 앵커가 없을 때의 감점.
    # 슬랩 단독에는 격자 해상도 개념이 없어 `multires_penalty` 로는 표현할 수 없다.
    coarse_volume_model: bool = False
    # 두께를 **아예 못 잰** 경우. 관측 높이가 깊이 노이즈에 묻혀 clamp 로 채운 것과
    # "조금 어긋나 clamp 된" 것은 전혀 다르다 — 전자는 부피가 통째로 가정이다.
    thickness_unmeasured: bool = False

    def tier(self) -> str:
        if not self.quality_ok:
            return "low"
        # 두께가 가정뿐이면 무게도 가정뿐이다. medium 으로 내보내면 사용자는
        # "쟀다"고 읽는다 — 실측(도련님 반지): 관측 높이 p95 0.87mm 가 깊이
        # 노이즈 RMSE 0.72mm 와 사실상 같아 두께를 못 쟀는데 tier=medium 이 나갔다.
        if self.thickness_unmeasured:
            return "low"
        score = 2
        if self.multires_penalty:
            score -= 1
        if not self.scale_tight:
            score -= 1
        if self.coarse_volume_model:
            score -= 1
        score = max(0, score)
        if self.precision_boost and score < 2:
            score += 1
        # 부피 모델이 상한 근사이거나 두께를 관측하지 못했으면 `precision_boost` 로도
        # `high` 로 올리지 않는다. precision_boost 는 **카메라 포즈·스케일** 품질
        # 신호일 뿐, 두께 미관측을 상쇄하지 못한다.
        # (실제로 폰 사진 + 기울인 카드면 boost 가 켜져 tier=high 가 됐는데,
        #  같은 결과의 부피 불확실성은 σ≈0.5 였다 — tier 와 범위가 서로 모순됐다)
        if self.coarse_volume_model:
            score = min(score, 1)
        return ["low", "medium", "high"][score]

    def pct(self) -> float:
        t = self.tier()
        return {"high": 82.0, "medium": 58.0, "low": 28.0}[t]


def apply_prior_demotion(mass_g: float, product_k: str, tier: str) -> str:
    lo, hi = PRIOR_MASS_G.get(product_k.lower(), PRIOR_MASS_G["other"])
    if mass_g < lo * 0.35 or mass_g > hi * 2.5:
        order = {"high": "medium", "medium": "low", "low": "low"}
        return order.get(tier, "low")
    return tier


def mass_range_heuristic(mass: float, tier: str) -> tuple[float, float, float]:
    """
    MIN / EST / MAX — tier 만 보는 고정폭.

    다뷰(v1) 경로용. 측정 가능한 불확실성이 없어 상수를 쓴다.
    단일사진(v2) 경로는 `mass_range_from_uncertainty` 를 쓴다.
    """
    if tier == "high":
        w = 0.12
    elif tier == "medium":
        w = 0.22
    else:
        w = 0.38
    return mass * (1 - w), mass, mass * (1 + w)


# tier 자체가 담고 있는 잔여 불확실성(모델·세그 품질 등)
_TIER_BASE_SIGMA = {"high": 0.10, "medium": 0.18, "low": 0.30}
_MAX_RANGE_WIDTH = 0.75


def volume_relative_sigma(
    *,
    anchor_used: bool,
    depth_rmse_mm: float | None,
    reference_distance_mm: float | None,
    thickness_assumed: bool,
    weak_volume_model: bool,
) -> float:
    """
    부피의 상대 표준편차 (§15.1 불확실성 전파).

    두 성분을 직교 합성한다.

    1. **스케일**: 길이 오차는 부피에서 3배가 된다. 앵커가 있으면 홀드아웃
       depth RMSE 를 기준 거리로 나눠 실제 측정값을 쓰고, 없으면 보수적 상수.
    2. **두께**: 한 장으로 두께를 못 봐서 클램프했다면 그 두께는 관측이 아니라
       가정이므로 큰 항을 더한다.
    """
    if anchor_used and depth_rmse_mm and reference_distance_mm:
        rel_scale = abs(depth_rmse_mm) / max(abs(reference_distance_mm), 1e-6)
    elif anchor_used:
        rel_scale = 0.05
    else:
        rel_scale = 0.15

    sigma = 3.0 * rel_scale
    if thickness_assumed:
        sigma = math.hypot(sigma, 0.50)
    elif weak_volume_model:
        sigma = math.hypot(sigma, 0.30)
    return sigma


def mass_range_from_uncertainty(
    mass: float, tier: str, volume_sigma: float
) -> tuple[float, float, float]:
    """측정된 부피 불확실성 + tier 기본항 → MIN / EST / MAX."""
    base = _TIER_BASE_SIGMA.get(tier, _TIER_BASE_SIGMA["low"])
    w = min(_MAX_RANGE_WIDTH, math.hypot(base, max(0.0, volume_sigma)))
    return mass * (1 - w), mass, mass * (1 + w)
