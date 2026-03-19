from __future__ import annotations

import statistics
from typing import Any

from app.config import Settings
from app.constants import VIEW_ORDER, VOXEL_GRID_N
from app.models.schemas import JobInputRecord, JobResult, MassRange
from app.pipeline import confidence as conf_mod
from app.pipeline import hollow
from app.pipeline import voxel as voxel_mod
from app.pipeline.card import CardGeometry, compute_card_geometry
from app.pipeline.exceptions import PipelineError
from app.pipeline.geometry_g1 import jewel_bbox_uv_mm
from app.pipeline.ingest import bytes_to_bgr, collect_exif
from app.pipeline.quality_gate import check_image_quality
from app.pipeline.segment import build_jewel_mask


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

    for view in VIEW_ORDER:
        raw = images[view]
        exif_meta[view] = collect_exif(raw)
        bgr = bytes_to_bgr(raw)
        check_image_quality(bgr, settings, view)
        card = compute_card_geometry(bgr, view)
        cards[view] = card
        sigmas[view] = card.sigma_mm_per_px
        mask = build_jewel_mask(bgr, card, settings, view, job_id=job_id)
        masks[view] = mask
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
    mass = hollow.mass_g(V_adj, inp.metal, inp.purity)

    precision_views = sum(1 for v in VIEW_ORDER if cards[v].precision_pose_candidate)
    precision_boost = precision_views >= 3

    cstate = conf_mod.ConfidenceState(
        multires_penalty=vol_est.multires_penalty,
        scale_tight=True,
        quality_ok=True,
        precision_boost=precision_boost,
    )
    tier = cstate.tier()
    tier = conf_mod.apply_prior_demotion(mass, inp.product_k, tier)
    pct = conf_mod.ConfidenceState(
        multires_penalty=vol_est.multires_penalty,
        scale_tight=True,
        quality_ok=(tier != "low"),
        precision_boost=precision_boost,
    ).pct()
    if tier == "low":
        pct = min(pct, 35.0)
    if precision_boost and tier == "high":
        pct = min(92.0, pct + 3.0)

    mn, est, mx = conf_mod.mass_range_heuristic(mass, tier)

    result = JobResult(
        algorithm_version=settings.algorithm_version,
        V_hull_mm3=round(vol_est.V_hull_mm3, 4),
        V_adj_mm3=round(V_adj, 4),
        mass_est_g=round(mass, 4),
        confidence_tier=tier,
        confidence_pct=round(pct, 2),
        mass_range=MassRange(min_g=round(mn, 4), estimate_g=round(est, 4), max_g=round(mx, 4)),
        meta={
            "hollow": {"alpha_k": alpha_k, "beta_k": beta_k, "product_k": inp.product_k},
            "sigmas_mm_per_px": {k: round(v, 6) for k, v in sigmas.items()},
            "exif": exif_meta,
            "volume_model": vol_est.volume_model,
            "precision": {
                "views_with_pose_candidate": precision_views,
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
