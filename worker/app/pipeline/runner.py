from __future__ import annotations

import statistics
from typing import Any

from app.config import Settings
from app.constants import SANITY_MAX_MASS_G_BY_PRODUCT, VIEW_ORDER, VOXEL_GRID_N
from app.models.schemas import JobInputRecord, JobResult, MassRange
from app.pipeline import confidence as conf_mod
from app.pipeline import hollow
from app.pipeline import voxel as voxel_mod
from app.pipeline.card import CardGeometry, compute_card_geometry
from app.pipeline.exceptions import PipelineError
from app.pipeline.geometry_g1 import jewel_bbox_uv_mm
from app.pipeline import jewel_layout as jewel_layout_mod
from app.pipeline.ingest import bytes_to_bgr, collect_exif
from app.pipeline.quality_gate import check_image_quality
from app.pipeline.segment import build_jewel_mask


def _sanity_mass_cap_g(product_k: str, settings: Settings) -> float:
    pk = product_k.lower()
    per = SANITY_MAX_MASS_G_BY_PRODUCT.get(pk, SANITY_MAX_MASS_G_BY_PRODUCT["other"])
    return min(float(settings.sanity_max_mass_g), per)


def _check_sigma_consistency(sigmas: dict[str, float], ratio: float, settings: Settings) -> None:
    vals = list(sigmas.values())
    med = statistics.median(vals)
    if med <= 0:
        raise PipelineError("ERR_SCALE_MISMATCH", "Invalid sigma median", retry_step=None)
    for v, s in sigmas.items():
        if abs(s - med) / med > ratio:
            raise PipelineError(
                "ERR_SCALE_MISMATCH",
                f"sigma view {v}={s:.5f} vs median {med:.5f}",
                retry_step=v,
            )


def run_pipeline(
    job_id: str,
    inp: JobInputRecord,
    images: dict[str, bytes],
    settings: Settings,
) -> dict[str, Any]:
    exif_meta: dict[str, dict] = {}
    sigmas: dict[str, float] = {}
    bboxes: dict[str, tuple[float, float, float, float]] = {}
    bboxes_coarse: dict[str, tuple[float, float, float, float]] = {}
    masks: dict[str, Any] = {}
    cards: dict[str, CardGeometry] = {}
    placement_by_view: dict[str, str] = {}

    for view in VIEW_ORDER:
        raw = images[view]
        exif_meta[view] = collect_exif(raw)
        bgr = bytes_to_bgr(raw)
        check_image_quality(bgr, settings, view)
        card = compute_card_geometry(bgr, view, settings)
        cards[view] = card
        sigmas[view] = card.sigma_mm_per_px
        mask, seg_meta = build_jewel_mask(bgr, card, settings, view, job_id=job_id)
        masks[view] = mask
        placement_by_view[view] = seg_meta.get("placement_mode", "subtract_card")
        u0, u1, v0, v1 = jewel_bbox_uv_mm(mask, card, view)
        bboxes[view] = (u0, u1, v0, v1)
        u_c = 0.5 * (u0 + u1)
        v_c = 0.5 * (v0 + v1)
        hu = 0.5 * (u1 - u0) * 0.85
        hv = 0.5 * (v1 - v0) * 0.85
        bboxes_coarse[view] = (u_c - hu, u_c + hu, v_c - hv, v_c + hv)

    _check_sigma_consistency(sigmas, settings.scale_mismatch_ratio, settings)

    view_items = [(v, masks[v], cards[v]) for v in VIEW_ORDER]
    vol_est = voxel_mod.estimate_volume(
        bboxes,
        bboxes_coarse,
        settings.voxel_penalty_resolution_ratio,
        use_carving=settings.use_voxel_carve,
        view_items=view_items,
        grid_n=VOXEL_GRID_N,
    )
    V_adj, alpha_k, beta_k = hollow.adjusted_volume_mm3(vol_est.V_hull_mm3, inp.product_k)
    V_adj_after_hollow = V_adj
    r_max, r_min_side = jewel_layout_mod.jewel_to_card_size_ratios(masks, cards)
    layout_mult, layout_detail = jewel_layout_mod.layout_volume_multiplier(
        inp.product_k, r_max, r_min_side
    )
    V_adj = V_adj * layout_mult
    mass = hollow.mass_g(V_adj, inp.metal, inp.purity)

    fallback_views = [v for v in VIEW_ORDER if cards[v].used_fallback_quad]
    mass_cap_g = _sanity_mass_cap_g(inp.product_k, settings)
    implausible_mass = mass > mass_cap_g

    precision_views = sum(1 for v in VIEW_ORDER if cards[v].precision_pose_candidate)
    precision_boost = precision_views >= 3

    # 카드 폴백·비현실적 무게 → 신뢰도 하한(§14.1 견적 게이팅과 정합)
    quality_ok = (len(fallback_views) == 0) and (not implausible_mass)
    scale_tight = len(fallback_views) == 0

    cstate = conf_mod.ConfidenceState(
        multires_penalty=vol_est.multires_penalty,
        scale_tight=scale_tight,
        quality_ok=quality_ok,
        precision_boost=precision_boost,
    )
    tier = cstate.tier()
    tier = conf_mod.apply_prior_demotion(mass, inp.product_k, tier)
    pct = conf_mod.ConfidenceState(
        multires_penalty=vol_est.multires_penalty,
        scale_tight=scale_tight,
        quality_ok=(tier != "low"),
        precision_boost=precision_boost,
    ).pct()
    if tier == "low":
        pct = min(pct, 35.0)
    if precision_boost and tier == "high":
        pct = min(92.0, pct + 3.0)

    mn, est, mx = conf_mod.mass_range_heuristic(mass, tier)

    suppress_mass_display = implausible_mass
    mass_range_out: MassRange | None = MassRange(min_g=round(mn, 4), estimate_g=round(est, 4), max_g=round(mx, 4))
    if suppress_mass_display:
        mass_range_out = None

    sanity_meta: dict[str, Any] = {
        "suppress_mass_display": suppress_mass_display,
        "implausible_mass": implausible_mass,
        "sanity_mass_cap_g": round(mass_cap_g, 4),
        "used_card_fallback_views": fallback_views,
        "warnings": [],
    }
    if fallback_views:
        sanity_meta["warnings"].append(
            "일부 뷰에서 카드를 자동으로 가정했습니다. 카드가 선명히 보이는 사진으로 다시 시도해 주세요."
        )
    if implausible_mass:
        sanity_meta["warnings"].append(
            f"추정 무게가 비현실적으로 큽니다(상한 {mass_cap_g:.0f} g 초과). "
            "귀금속·카드 프로토콜에 맞는 촬영인지 확인해 주세요."
        )
        sanity_meta["raw_mass_est_g"] = round(mass, 4)

    if layout_detail.get("applied"):
        b = layout_detail.get("bucket", "?")
        reff = layout_detail.get("ratio_effective")
        m = layout_detail.get("layout_volume_mult")
        sanity_meta["warnings"].append(
            f"카드 대비 실루엣 부피 보정 적용: 유효비≈{reff}, 구간={b}, 부피×{m}. "
            "형태(귀걸이/반지 등)와 촬영에 맞게 `constants.py` Hollow·layout 계수를 실측으로 조정하세요."
        )

    result = JobResult(
        algorithm_version=settings.algorithm_version,
        V_hull_mm3=round(vol_est.V_hull_mm3, 4),
        V_adj_mm3=round(V_adj, 4),
        mass_est_g=round(mass, 4),
        confidence_tier=tier,
        confidence_pct=round(pct, 2),
        mass_range=mass_range_out,
        meta={
            "segmentation": {"per_view_placement": placement_by_view},
            "hollow": {
                "alpha_k": alpha_k,
                "beta_k": beta_k,
                "product_k": inp.product_k,
                "V_adj_mm3_after_hollow_before_layout": round(V_adj_after_hollow, 4),
                "layout_correction": layout_detail,
            },
            "sigmas_mm_per_px": {k: round(v, 6) for k, v in sigmas.items()},
            "exif": exif_meta,
            "volume_model": vol_est.volume_model,
            "sanity": sanity_meta,
            "precision": {
                "views_with_pose_candidate": precision_views,
                "used_fallback_quad_views": fallback_views,
                "per_view": {
                    v: {
                        "candidate": cards[v].precision_pose_candidate,
                        "solutions": cards[v].precision_solution_count,
                    }
                    for v in VIEW_ORDER
                },
            },
        },
    )
    return result.model_dump()
