from __future__ import annotations

from app.constants import HOLLOW_ALPHA_BETA, RHO_G_CM3


def adjusted_volume_mm3(V_hull: float, product_k: str) -> tuple[float, float, float]:
    """Returns V_adj, alpha, beta."""
    alpha, beta = HOLLOW_ALPHA_BETA.get(product_k.lower(), HOLLOW_ALPHA_BETA["other"])
    return alpha * V_hull + beta, alpha, beta


def mass_g(V_adj_mm3: float, metal: str, purity: str) -> float:
    key = (metal.lower(), purity.lower())
    rho = RHO_G_CM3.get(key)
    if rho is None:
        # fallback 18k gold
        rho = RHO_G_CM3[("gold", "18k")]
    cm3 = V_adj_mm3 / 1000.0
    return rho * cm3
