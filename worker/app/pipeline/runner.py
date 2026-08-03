"""
파이프라인 실행기 — `capture_mode` 로 두 경로를 분기한다.

- `single`   : v2 기본. 사진 1장 → 검출 → 분할 → 깊이 → 앵커 스케일 융합 → 2.5D 부피
- `multiview`: v1 계승. 5뷰 → 카드 σ → 실루엣 → 복셀 카빙

스펙: `archimedes-v2-single-photo.mdc` §1(파이프라인)·§4(부피)·§5(다뷰 모드)
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from app.config import Settings
from app.constants import SANITY_MAX_MASS_G_BY_PRODUCT, VIEW_ORDER, VOXEL_GRID_N
from app.models.schemas import JobInputRecord, JobResult, MassRange
from app.pipeline import confidence as conf_mod
from app.pipeline import hollow
from app.pipeline import jewel_layout as jewel_layout_mod
from app.pipeline import voxel as voxel_mod
from app.pipeline.backends import get_depth_estimator, get_detector, get_segmenter
from app.pipeline.backends.types import Detection
from app.pipeline.camera import intrinsics_from_exif
from app.pipeline.card import CardGeometry, compute_card_geometry, try_compute_card_geometry
from app.pipeline.exceptions import PipelineError
from app.pipeline.geometry_g1 import jewel_bbox_uv_mm
from app.pipeline.ingest import bytes_to_bgr, collect_exif
from app.pipeline.jewel_mask import refine_jewel_mask
from app.pipeline.quality_gate import check_image_quality
from app.pipeline.reconstruct import reconstruct_from_depth
from app.pipeline.scale_fusion import fuse_scale
from app.pipeline.segment import build_jewel_mask

SINGLE_VIEW_KEY = "front"


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
                error_severity="soft",
                suggested_action="retry_one_view",
            )


def run_pipeline(
    job_id: str,
    inp: JobInputRecord,
    images: dict[str, bytes],
    settings: Settings,
) -> dict[str, Any]:
    if inp.capture_mode == "multiview":
        return _run_multiview(job_id, inp, images, settings)
    return _run_single(job_id, inp, images, settings)


# ══════════════════════════ v2 — 단일 사진 ══════════════════════════


def _card_bounding_rect(card: CardGeometry) -> tuple[int, int, int, int]:
    q = np.asarray(card.quad_px, dtype=np.float64)
    return (
        int(np.floor(q[:, 0].min())),
        int(np.floor(q[:, 1].min())),
        int(np.ceil(q[:, 0].max())),
        int(np.ceil(q[:, 1].max())),
    )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _select_jewelry_box(
    dets: list[Detection], card: CardGeometry | None
) -> tuple[Detection | None, dict[str, Any]]:
    """
    검출 결과에서 귀금속 박스를 고른다.

    카드가 검출됐다면 **카드 자신인 박스**(카드 외접 사각형과 IoU 가 높은 것)를 뺀다.
    남는 게 없으면 `None` 을 돌려 분할을 프레임 전체로 돌린다 — 잘못된 박스로
    실루엣을 잘라 버리는 것보다 안전하다.
    """
    meta: dict[str, Any] = {"n_detections": len(dets)}
    if not dets:
        meta["selected"] = None
        return None, meta

    candidates = dets
    if card is not None:
        card_rect = _card_bounding_rect(card)
        candidates = [d for d in dets if _iou(d.box_xyxy, card_rect) < 0.6]
        meta["dropped_as_card"] = len(dets) - len(candidates)
    if not candidates:
        meta["selected"] = None
        return None, meta

    best = max(candidates, key=lambda d: d.area())
    meta["selected"] = {"box": list(best.box_xyxy), "score": round(best.score, 5), "label": best.label}
    return best, meta


def _run_single(
    job_id: str,
    inp: JobInputRecord,
    images: dict[str, bytes],
    settings: Settings,
) -> dict[str, Any]:
    raw = images.get(SINGLE_VIEW_KEY)
    if raw is None:
        raise PipelineError("ERR_VIEWS", "Single-photo job has no image", retry_step=None)

    exif = collect_exif(raw)
    bgr = bytes_to_bgr(raw, exif.get("orientation"))
    check_image_quality(bgr, settings, SINGLE_VIEW_KEY)
    h, w = bgr.shape[:2]
    K = intrinsics_from_exif(exif, w, h)

    # 카드는 v2 에서 **선택적 앵커** — 없으면 깊이 단독 경로로 간다
    card = try_compute_card_geometry(bgr, SINGLE_VIEW_KEY, settings)

    detector = get_detector(settings)
    box, det_meta = _select_jewelry_box(detector.detect(bgr), card)
    det_meta["backend"] = detector.name

    segmenter = get_segmenter(settings)
    seg = segmenter.segment(bgr, box)
    mask, mask_meta = refine_jewel_mask(seg.mask, card, SINGLE_VIEW_KEY)
    mask_meta["backend"] = segmenter.name

    depth_est = get_depth_estimator(settings)
    dmap = depth_est.estimate(bgr)
    fusion = fuse_scale(dmap, K, card, require_anchor=settings.require_anchor)

    rec = reconstruct_from_depth(
        mask,
        fusion.depth_mm,
        K,
        inp.product_k,
        support_plane=fusion.support_plane,
        valid=dmap.valid_mask(),
    )

    V_adj, alpha_k, beta_k = hollow.adjusted_volume_depth_mm3(rec.volume_mm3, inp.product_k)
    mass = hollow.mass_g(V_adj, inp.metal, inp.purity)

    mass_cap_g = _sanity_mass_cap_g(inp.product_k, settings)
    implausible_mass = mass > mass_cap_g

    # ── 신뢰도 입력 매핑 ──
    scale_tight = fusion.anchor_used and not fusion.ill_conditioned
    thickness_assumed = rec.thickness_clamp is not None
    weak_model = rec.method != "height_field"
    rel_rmse = None
    if fusion.depth_rmse_mm is not None and fusion.card_distance_mm:
        rel_rmse = fusion.depth_rmse_mm / fusion.card_distance_mm
    depth_penalty = rel_rmse is not None and rel_rmse > settings.depth_rmse_penalty_ratio

    cstate = conf_mod.ConfidenceState(
        multires_penalty=depth_penalty,
        scale_tight=scale_tight,
        quality_ok=not implausible_mass,
        precision_boost=K.is_reliable and scale_tight,
        coarse_volume_model=weak_model or thickness_assumed,
    )
    tier = conf_mod.apply_prior_demotion(mass, inp.product_k, cstate.tier())
    pct = conf_mod.ConfidenceState(
        multires_penalty=depth_penalty,
        scale_tight=scale_tight,
        quality_ok=(tier != "low"),
        precision_boost=K.is_reliable and scale_tight,
        coarse_volume_model=weak_model or thickness_assumed,
    ).pct()
    if tier == "low":
        pct = min(pct, 35.0)

    vol_sigma = conf_mod.volume_relative_sigma(
        anchor_used=fusion.anchor_used,
        depth_rmse_mm=fusion.depth_rmse_mm,
        reference_distance_mm=fusion.card_distance_mm,
        thickness_assumed=thickness_assumed,
        weak_volume_model=weak_model,
    )
    mn, est, mx = conf_mod.mass_range_from_uncertainty(mass, tier, vol_sigma)

    warnings: list[str] = []
    if not fusion.anchor_used:
        warnings.append(
            "신용카드(기준물)를 찾지 못해 깊이 모델 스케일만 사용했습니다. "
            "카드를 귀금속과 같은 바닥에 함께 두고 찍으면 정확도가 크게 올라갑니다."
        )
    if fusion.ill_conditioned:
        warnings.append(
            "카드가 카메라와 거의 평행해 거리 변화가 작습니다. 살짝 각도를 주고 찍어 주세요."
        )
    if thickness_assumed:
        warnings.append(
            f"두께를 관측하지 못해 제품 기준값({rec.h_mean_mm:.1f} mm)으로 가정했습니다. "
            "실제 무게와 차이가 클 수 있습니다."
        )
    if not K.is_reliable:
        warnings.append("사진에 초점거리 정보(EXIF)가 없어 카메라 값을 추정했습니다.")
    if implausible_mass:
        warnings.append(
            f"추정 무게가 비현실적으로 큽니다(상한 {mass_cap_g:.0f} g 초과). 촬영을 다시 확인해 주세요."
        )

    sanity_meta: dict[str, Any] = {
        "suppress_mass_display": implausible_mass,
        "implausible_mass": implausible_mass,
        "sanity_mass_cap_g": round(mass_cap_g, 4),
        "used_card_fallback_views": [],
        "warnings": warnings,
    }
    if implausible_mass:
        sanity_meta["raw_mass_est_g"] = round(mass, 4)

    # 저하 사유를 기계가 읽을 수 있게 남긴다(프런트 재촬영 유도용).
    # 단일사진은 뷰가 하나뿐이라 재시도 액션이 "retake_photo" 다.
    degraded_reasons: list[str] = []
    if not fusion.anchor_used:
        degraded_reasons.append("no_anchor")
    if fusion.ill_conditioned:
        degraded_reasons.append("anchor_ill_conditioned")
    if thickness_assumed:
        degraded_reasons.append(f"thickness_clamped:{rec.thickness_clamp}")
    if weak_model:
        degraded_reasons.append(f"weak_volume_model:{rec.method}")
    if not K.is_reliable:
        degraded_reasons.append("intrinsics_fallback")
    if depth_penalty:
        degraded_reasons.append("depth_rmse_high")
    if implausible_mass:
        degraded_reasons.append("implausible_mass")

    result = JobResult(
        algorithm_version=settings.algorithm_version,
        V_hull_mm3=round(rec.volume_mm3, 4),
        V_adj_mm3=round(V_adj, 4),
        mass_est_g=round(mass, 4),
        confidence_tier=tier,
        confidence_pct=round(pct, 2),
        mass_range=(
            None
            if implausible_mass
            else MassRange(min_g=round(mn, 4), estimate_g=round(est, 4), max_g=round(mx, 4))
        ),
        meta={
            "capture_mode": "single",
            "workflow": {
                "error_severity": "soft" if tier == "low" else "none",
                "suggested_action": (
                    "retake_photo" if degraded_reasons else "continue_low_confidence"
                ),
                "retry_views": [SINGLE_VIEW_KEY] if degraded_reasons else [],
                "degraded_reasons": degraded_reasons,
            },
            "volume_model": rec.method,
            "camera": K.as_meta(),
            "detection": det_meta,
            "segmentation": mask_meta,
            "depth": {"backend": dmap.meta.get("backend"), "kind": dmap.kind.value},
            "scale_fusion": fusion.as_meta(),
            "reconstruction": rec.as_meta(),
            "uncertainty": {
                "volume_relative_sigma": round(vol_sigma, 4),
                "depth_rmse_over_distance": round(rel_rmse, 5) if rel_rmse is not None else None,
            },
            "hollow": {
                "alpha_k": alpha_k,
                "beta_k": beta_k,
                "product_k": inp.product_k,
                "table": "depth",
            },
            "exif": {SINGLE_VIEW_KEY: exif},
            "sanity": sanity_meta,
        },
    )
    return result.model_dump()


# ══════════════════════════ v1 — 5뷰 다뷰 모드 ══════════════════════════


def _run_multiview(
    job_id: str,
    inp: JobInputRecord,
    images: dict[str, bytes],
    settings: Settings,
) -> dict[str, Any]:
    exif_meta: dict[str, dict] = {}
    sigmas: dict[str, float] = {}
    bboxes: dict[str, tuple[float, float, float, float]] = {}
    masks: dict[str, Any] = {}
    cards: dict[str, CardGeometry] = {}
    placement_by_view: dict[str, str] = {}

    for view in VIEW_ORDER:
        raw = images[view]
        exif_meta[view] = collect_exif(raw)
        # EXIF orientation 을 화소에 반영해야 뷰↔월드 축 매핑이 성립한다(v2 §0.4 #4)
        bgr = bytes_to_bgr(raw, exif_meta[view].get("orientation"))
        check_image_quality(bgr, settings, view)
        card = compute_card_geometry(bgr, view, settings)
        cards[view] = card
        sigmas[view] = card.sigma_mm_per_px
        mask, seg_meta = build_jewel_mask(bgr, card, settings, view, job_id=job_id)
        masks[view] = mask
        placement_by_view[view] = seg_meta.get("placement_mode", "subtract_card")
        bboxes[view] = jewel_bbox_uv_mm(mask, card, view)

    _check_sigma_consistency(sigmas, settings.scale_mismatch_ratio, settings)

    view_items = [(v, masks[v], cards[v]) for v in VIEW_ORDER]
    vol_est = voxel_mod.estimate_volume(
        bboxes,
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
    coarse_model = vol_est.volume_model != "voxel_carve"

    cstate = conf_mod.ConfidenceState(
        multires_penalty=vol_est.multires_penalty,
        scale_tight=scale_tight,
        quality_ok=quality_ok,
        precision_boost=precision_boost,
        coarse_volume_model=coarse_model,
    )
    tier = cstate.tier()
    tier = conf_mod.apply_prior_demotion(mass, inp.product_k, tier)
    pct = conf_mod.ConfidenceState(
        multires_penalty=vol_est.multires_penalty,
        scale_tight=scale_tight,
        quality_ok=(tier != "low"),
        precision_boost=precision_boost,
        coarse_volume_model=coarse_model,
    ).pct()
    if tier == "low":
        pct = min(pct, 35.0)
    if precision_boost and tier == "high":
        pct = min(92.0, pct + 3.0)

    mn, est, mx = conf_mod.mass_range_heuristic(mass, tier)

    suppress_mass_display = implausible_mass
    mass_range_out: MassRange | None = MassRange(
        min_g=round(mn, 4), estimate_g=round(est, 4), max_g=round(mx, 4)
    )
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

    degraded_reasons: list[str] = []
    if fallback_views:
        degraded_reasons.extend([f"card_fallback:{v}" for v in fallback_views])
    if implausible_mass:
        degraded_reasons.append("implausible_mass")
    if vol_est.multires_penalty:
        degraded_reasons.append("multires_penalty")

    retry_views = sorted(set(fallback_views))[:2]

    result = JobResult(
        algorithm_version=settings.algorithm_version,
        V_hull_mm3=round(vol_est.V_hull_mm3, 4),
        V_adj_mm3=round(V_adj, 4),
        mass_est_g=round(mass, 4),
        confidence_tier=tier,
        confidence_pct=round(pct, 2),
        mass_range=mass_range_out,
        meta={
            "capture_mode": "multiview",
            "workflow": {
                "error_severity": "soft" if tier == "low" else "none",
                "suggested_action": "retry_one_view" if retry_views else "continue_low_confidence",
                "retry_views": retry_views,
                "degraded_reasons": degraded_reasons,
            },
            "segmentation": {"per_view_placement": placement_by_view},
            "hollow": {
                "alpha_k": alpha_k,
                "beta_k": beta_k,
                "product_k": inp.product_k,
                "table": "multiview",
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
