from __future__ import annotations

from app.constants import (
    HOLLOW_ALPHA_BETA,
    HOLLOW_ALPHA_BETA_DEPTH,
    MATERIALS,
    METAL_ALIASES,
    PURITY_ALIASES,
    Material,
)
from app.pipeline.exceptions import PipelineError


def adjusted_volume_mm3(
    V_hull: float, product_k: str, *, table: dict[str, tuple[float, float]] | None = None
) -> tuple[float, float, float]:
    """
    V_adj = α_k · V_hull + β_k. Returns (V_adj, alpha, beta).

    `table` 로 경로별 α 표를 고른다. 다뷰(v1) Visual Hull 과 단일사진(v2) 2.5D 부피는
    과대추정 성격이 완전히 달라 같은 α 를 쓰면 안 된다.
    """
    t = table if table is not None else HOLLOW_ALPHA_BETA
    alpha, beta = t.get(product_k.lower(), t["other"])
    return alpha * V_hull + beta, alpha, beta


def adjusted_volume_depth_mm3(V_25d: float, product_k: str) -> tuple[float, float, float]:
    """단일사진(depth) 경로 전용 — `HOLLOW_ALPHA_BETA_DEPTH`."""
    return adjusted_volume_mm3(V_25d, product_k, table=HOLLOW_ALPHA_BETA_DEPTH)


def resolve_material(metal: str, purity: str) -> Material:
    """
    (금속, 함량) 문자열 → `Material`.

    미지원 조합은 **조용히 18K 금으로 폴백하지 않는다.** 이전 구현은 그렇게 해서
    백금·22K 요청이 말없이 다른 밀도로 계산됐다(무게가 통째로 틀림).
    """
    m_raw = (metal or "").strip().lower()
    p_raw = (purity or "").strip().lower()
    m = METAL_ALIASES.get(m_raw)
    if m is None:
        raise PipelineError(
            "ERR_UNSUPPORTED_MATERIAL",
            f"Unsupported metal: {metal!r}. supported={sorted(set(METAL_ALIASES.values()))}",
        )
    p = PURITY_ALIASES.get(m, {}).get(p_raw)
    if p is None:
        allowed = sorted({v for v in PURITY_ALIASES.get(m, {}).values()})
        raise PipelineError(
            "ERR_UNSUPPORTED_MATERIAL",
            f"Unsupported purity {purity!r} for metal {m!r}. supported={allowed}",
        )
    mat = MATERIALS.get((m, p))
    if mat is None:  # 별칭 표와 물성 표가 어긋난 경우 — 배포 전 잡혀야 한다
        raise PipelineError(
            "ERR_UNSUPPORTED_MATERIAL", f"No density row for ({m!r}, {p!r})"
        )
    return mat


def rho_g_cm3(metal: str, purity: str) -> float:
    return resolve_material(metal, purity).rho_g_cm3


def mass_g(V_adj_mm3: float, metal: str, purity: str) -> float:
    """V_adj(mm³) × ρ(g/cm³) → g. 1 cm³ = 1000 mm³."""
    rho = rho_g_cm3(metal, purity)
    return rho * (V_adj_mm3 / 1000.0)
