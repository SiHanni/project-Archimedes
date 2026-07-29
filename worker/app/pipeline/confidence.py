from __future__ import annotations

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

    def tier(self) -> str:
        if not self.quality_ok:
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
    """MIN / EST / MAX — §15.1 style stub."""
    if tier == "high":
        w = 0.12
    elif tier == "medium":
        w = 0.22
    else:
        w = 0.38
    return mass * (1 - w), mass, mass * (1 + w)
