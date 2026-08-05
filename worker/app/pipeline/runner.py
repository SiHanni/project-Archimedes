"""
파이프라인 실행기 — `capture_mode` 로 두 경로를 분기한다.

- `single`   : v2 기본. 사진 1장 → 검출 → 분할 → 깊이 → 앵커 스케일 융합 → 2.5D 부피
- `multiview`: v1 계승. 5뷰 → 카드 σ → 실루엣 → 복셀 카빙

스펙: `archimedes-v2-single-photo.mdc` §1(파이프라인)·§4(부피)·§5(다뷰 모드)
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

import cv2
import numpy as np

from app.config import Settings
from app.constants import (
    FLAT_PRODUCTS,
    SANITY_MAX_MASS_G_BY_PRODUCT,
    VIEW_ORDER,
    VOLUME_UNMEASURABLE_PRODUCTS,
    VOXEL_GRID_N,
)
from app.models.schemas import JobInputRecord, JobResult, MassRange
from app.pipeline import confidence as conf_mod
from app.pipeline import hollow
from app.pipeline import jewel_layout as jewel_layout_mod
from app.pipeline import voxel as voxel_mod
from app.pipeline.backends import (
    get_depth_estimator,
    get_detector,
    get_ocr_reader,
    get_segmenter,
)
from app.pipeline.backends.types import Detection
from app.pipeline.camera import intrinsics_from_card, intrinsics_from_exif
from app.pipeline.card import (
    CardGeometry,
    card_edge_lengths_px,
    compute_card_geometry,
    try_compute_card_geometry,
)
from app.pipeline.exceptions import PipelineError
from app.pipeline.geometry_g1 import jewel_bbox_uv_mm
from app.pipeline.height_segment import segment_by_height
from app.pipeline.ingest import bytes_to_bgr, collect_exif
from app.pipeline.jewel_mask import card_roi_mask, refine_jewel_mask
from app.pipeline.matting import refine_with_grabcut
from app.pipeline.ocr import LabelReading, read_label
from app.pipeline.quality_gate import check_image_quality
from app.pipeline.reconstruct import reconstruct_from_depth
from app.pipeline.scale_fusion import fuse_scale
from app.pipeline.segment import build_jewel_mask
from app.pipeline.visualize import build_assets
from app.s3util import upload_object

log = logging.getLogger(__name__)

SINGLE_VIEW_KEY = "front"

# 측정 무게 / 표기 무게 가 이 배수를 넘으면 **몸체 전체가 순금은 아니다**
# (봉입형·금박·도금 — 금 자체는 따로 제련한 진짜 999 여도 몸체가 아크릴·수지).
# 순금이라면 두 값이 비슷해야 하고, 우리 부피 오차(§15.1 σ≈0.5)를 감안해도
# 3배를 넘기 어렵다.
SOLID_GOLD_RATIO_THRESHOLD = 3.0

# 누끼 정밀화가 넘어가면 안 되는 반경(카드 긴 변의 배수). 물체 탐색 ROI 보다
# 넉넉해야 물체 전체를 덮을 수 있고, 그래도 책상 전체를 먹지는 못한다.
MATTING_ROI_SPANS = 2.0


def _sanity_mass_cap_g(product_k: str, settings: Settings) -> float:
    pk = product_k.lower()
    per = SANITY_MAX_MASS_G_BY_PRODUCT.get(pk, SANITY_MAX_MASS_G_BY_PRODUCT["other"])
    return min(float(settings.sanity_max_mass_g), per)


# σ 가 중앙값 대비 이 배수를 넘으면 촬영 거리 차이로는 설명되지 않는다
# = 카드가 아닌 다른 사각형을 잡았다고 본다. 이때만 거절한다.
_SIGMA_ABSURD_RATIO = 4.0


def _sigma_consistency(sigmas: dict[str, float], warn_ratio: float) -> tuple[float, list[str]]:
    """
    뷰별 σ 의 흩어짐을 **보고**한다. 거절하지 않는다.

    σ 는 각 컷의 카드 크기로 따로 계산되므로 컷마다 다른 것이 **정상**이고,
    mm 변환에서 이미 보정된다. 실측상 손각대 5뷰는 카드 크기가 2배 가까이
    차이 나며, 그건 사용자가 잘못한 게 아니다.

    이전에는 ±8% 를 넘으면 거절했는데, 그 임계값은 σ 가 상수로 고정돼 있던
    시절의 값이라 한 번도 발동한 적이 없었다. σ 를 고치자 정상 촬영을 전부
    막아 버렸다.

    Returns (최대 상대편차, 경고 임계를 넘은 뷰 목록).
    """
    vals = list(sigmas.values())
    med = statistics.median(vals)
    if med <= 0:
        raise PipelineError(
            "ERR_SCALE_MISMATCH",
            "카드 스케일을 계산하지 못했습니다. 카드가 선명히 보이게 다시 찍어 주세요.",
            error_severity="soft",
            suggested_action="retry_one_view",
        )

    max_dev = 0.0
    outliers: list[str] = []
    for v, sv in sigmas.items():
        dev = abs(sv - med) / med
        max_dev = max(max_dev, dev)
        if sv / med > _SIGMA_ABSURD_RATIO or med / sv > _SIGMA_ABSURD_RATIO:
            raise PipelineError(
                "ERR_SCALE_MISMATCH",
                f"'{v}' 컷에서 카드가 아닌 다른 사각형을 잡은 것 같습니다"
                f"(다른 컷 대비 약 {sv / med:.1f}배). "
                "카드 전체가 잘리지 않고 또렷하게 나오도록 다시 찍어 주세요.",
                retry_step=v,
                error_severity="soft",
                suggested_action="retry_one_view",
            )
        if dev > warn_ratio:
            outliers.append(v)
    return max_dev, outliers


# 5뷰 기하가 깨졌을 때 단일사진 경로로 쓸 컷의 우선순위.
# 상단 뷰가 물체의 바닥 면적을 가장 잘 보여 준다.
_SINGLE_FALLBACK_ORDER = ("top", "front", "back", "left", "right")

# 다뷰 기하 실패 중 "한 장으로 재시도"가 의미 있는 코드
_GEOMETRY_FAILURE_CODES = frozenset({"ERR_VOLUME", "ERR_SCALE_MISMATCH"})


def run_pipeline(
    job_id: str,
    inp: JobInputRecord,
    images: dict[str, bytes],
    settings: Settings,
) -> dict[str, Any]:
    if inp.capture_mode != "multiview":
        return _run_single(job_id, inp, images, settings)

    try:
        return _run_multiview(job_id, inp, images, settings)
    except PipelineError as e:
        if e.code not in _GEOMETRY_FAILURE_CODES:
            raise
        # 5뷰 기하는 촬영 규약(정면/상/좌/우/후를 정확한 축으로)을 지켜야 성립한다.
        # 손각대로는 자주 깨지는데, 그때 사용자를 빈손으로 돌려보낼 이유가 없다.
        # 가장 쓸모 있는 한 컷으로 단일사진 경로를 돌리고 그 사실을 밝힌다.
        return _fallback_single_from_views(job_id, inp, images, settings, e)


def _fallback_single_from_views(
    job_id: str,
    inp: JobInputRecord,
    images: dict[str, bytes],
    settings: Settings,
    cause: PipelineError,
) -> dict[str, Any]:
    for view in _SINGLE_FALLBACK_ORDER:
        raw = images.get(view)
        if raw is None:
            continue
        try:
            out = _run_single(job_id, inp, {SINGLE_VIEW_KEY: raw}, settings)
        except PipelineError:
            continue

        meta = out.setdefault("meta", {})
        meta["capture_mode"] = "multiview_fallback_single"
        meta["multiview_fallback"] = {
            "used_view": view,
            "cause_code": cause.code,
            "cause_message": str(cause),
        }
        wf = meta.setdefault("workflow", {})
        reasons = list(wf.get("degraded_reasons") or [])
        reasons.append(f"multiview_geometry_failed:{cause.code}")
        wf["degraded_reasons"] = reasons
        wf["error_severity"] = "soft"
        wf["suggested_action"] = "retry_one_view"
        wf["retry_views"] = cause.retry_views or [view]

        sanity = meta.setdefault("sanity", {})
        warnings = list(sanity.get("warnings") or [])
        warnings.insert(
            0,
            f"5방향 사진의 각도가 서로 맞지 않아 '{view}' 한 장만으로 분석했습니다. "
            "정확도가 떨어지니, 각 슬롯에 맞는 각도로 다시 찍거나 "
            "'사진 1장' 모드를 이용해 주세요.",
        )
        sanity["warnings"] = warnings
        return out

    # 어느 컷으로도 안 되면 원래 기하 실패를 그대로 알린다
    raise cause


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


def _keep_near_card(dets: list[Detection], card: CardGeometry, spans: float = 1.2) -> list[Detection]:
    """카드 중심에서 카드 긴 변의 `spans` 배 안에 있는 검출만 남긴다."""
    cx = float(card.quad_px[:, 0].mean())
    cy = float(card.quad_px[:, 1].mean())
    long_px, _ = card_edge_lengths_px(card.quad_px)
    limit = long_px * spans
    near = []
    for d in dets:
        x0, y0, x1, y1 = d.box_xyxy
        bx, by = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if np.hypot(bx - cx, by - cy) <= limit:
            near.append(d)
    return near or dets


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

    # 카드는 v2 에서 **선택적 앵커** — 없으면 깊이 단독 경로로 간다
    card = try_compute_card_geometry(bgr, SINGLE_VIEW_KEY, settings, exif)

    # EXIF 에 초점거리가 없는 사진이 흔하다(실측: 도련님 사진 2장 모두 없음).
    # 그때 1.15·max(W,H) 로 추측하면 최대 46% 틀리고, 그 오차가 두께 →
    # 부피 → 무게로 그대로 간다. 카드가 있으면 **소실점으로 f 를 푼다.**
    K = (
        intrinsics_from_card(exif, card.quad_px, w, h)
        if card is not None
        else intrinsics_from_exif(exif, w, h)
    )

    depth_est = get_depth_estimator(settings)
    dmap = depth_est.estimate(bgr)
    fusion = fuse_scale(dmap, K, card, require_anchor=settings.require_anchor)

    # 세그멘테이션: **높이 우선**.
    # 실사용 사진은 책상 위라 프레임 절반이 키보드·상자다. 밝기·범용 배경제거는
    # 그걸 전부 전경으로 잡는다(실측: 마스크 24.5%, 404×252mm, 10.6kg).
    # 바닥 평면은 이미 정확히 알고 있으므로(홀드아웃 RMSE 0.6mm), 카드 근처에서
    # **평면 위로 솟은 것**을 물체로 본다. 색·조명·배경 무늬와 무관하다.
    det_meta: dict[str, Any] = {"backend": "skipped_height_segment"}
    mask = None
    mask_meta: dict[str, Any] = {}
    # 깊이 모델이 스스로 퇴화(상수 출력)라고 선언하면 높이 신호가 없다.
    # 그때 억지로 높이 세그를 돌리면 원근 때문에 화면 가장자리가 "솟은 것"으로
    # 잡힌다 — 실제 물체가 아니다.
    depth_usable = not dmap.meta.get("degenerate")
    # 골드바처럼 두께가 깊이 노이즈(~0.6mm)보다 얇은 제품은 **높이로 찾을 수 없다.**
    # 억지로 돌리면 못 찾거나 엉뚱한 걸 잡는다 — 외형 경로로 보낸다.
    is_flat = inp.product_k.lower() in FLAT_PRODUCTS
    if fusion.support_plane is not None and depth_usable and not is_flat:
        try:
            mask, mask_meta = segment_by_height(
                fusion.depth_mm,
                fusion.support_plane,
                K,
                card,
                depth_rmse_mm=fusion.depth_rmse_mm,
                side=settings.object_side,
            )
        except PipelineError as e:
            log.info("height segmentation failed (%s), falling back to appearance", e.code)

    if mask is None:
        # 앵커가 없거나 높이 세그가 실패·부적합한 경우 외형 기반으로 내려간다
        detector = get_detector(settings)
        dets = detector.detect(bgr)
        if card is not None:
            # 배경 잡동사니 제거 — §4 프로토콜상 물체는 **카드 옆**에 있다
            dets = _keep_near_card(dets, card)
        box, det_meta = _select_jewelry_box(dets, card)
        det_meta["backend"] = detector.name
        if detector.name == "stub":
            # 스텁은 "가장 큰 밝은 덩어리"일 뿐 귀금속 검출기가 아니다.
            # 그 박스로 분할을 가두면 엉뚱한 곳(카드 위 인쇄 등)만 보게 된다.
            # 실측: 박스가 카드 쪽에 잡혀 카드 글자를 물체로 읽었다("KB号").
            # ROI(카드 옆 반원)만으로 충분하므로 박스는 쓰지 않는다.
            det_meta["box_used"] = False
            box = None
        else:
            det_meta["box_used"] = box is not None
        segmenter = get_segmenter(settings)
        seg = segmenter.segment(bgr, box)
        mask, mask_meta = refine_jewel_mask(
            seg.mask, card, SINGLE_VIEW_KEY, side=settings.object_side
        )
        mask_meta["backend"] = segmenter.name

        # 외형 경로의 Otsu 는 금의 **반사로 빛나는 일부만** 잡는다(실측: 금괴
        # 상단 40%). 씨앗의 색 분포를 학습해 나머지를 끌어온다 — 누끼 품질이
        # 곧 계획서 Step 1 의 산출물이므로 여기서 한 번 정밀화한다.
        # 카드 자신과 **카드에서 너무 먼 곳**은 확정 배경으로 못박는다. 안 그러면
        # 색이 비슷한 책상이 통째로 딸려 온다(실측: 성장 6.3배, 231×182mm).
        # 반경은 탐색 ROI(카드 긴 변 1배)보다 넉넉히 잡는다 — 같은 반경을 쓰면
        # 물체 자체가 잘려 정밀화가 아무 일도 못 한다(실측: 성장 1.0배로 무력화).
        exclude = None
        if card is not None:
            exclude = cv2.bitwise_not(card_roi_mask(card, mask.shape, MATTING_ROI_SPANS))
            cv2.fillPoly(exclude, [np.asarray(card.quad_px, dtype=np.int32)], 255)
        mask, matte_meta = refine_with_grabcut(bgr, mask, exclude=exclude)
        mask_meta.update(matte_meta)
        # 정밀화로 마스크가 바뀌었으니 면적도 다시 적는다 — meta 는 실제로 쓴 마스크를 말해야 한다
        mask_meta["area_frac"] = round(float(np.count_nonzero(mask)) / float(h * w), 6)

    rec = reconstruct_from_depth(
        mask,
        fusion.depth_mm,
        K,
        inp.product_k,
        support_plane=fusion.support_plane,
        valid=dmap.valid_mask(),
        thickness_override_mm=inp.reference_thickness_mm,
    )

    # ── 각인 판독 ──
    # 골드바·봉입형은 두께가 마이크로미터라 부피로 못 잰다. 대신 함유량이
    # **제품에 새겨져 있다**. 측정할 수 없는 것을 추정하지 말고 적힌 것을 읽는다.
    # 전처리(CLAHE·업스케일)는 오히려 인식률을 떨어뜨렸다 — 금색 위 금색 각인이라
    # 대비를 키우면 문자 경계가 뭉개진다. **원본 컬러 크롭**을 그대로 쓴다.
    label: LabelReading = LabelReading()
    if is_flat:
        ys_m, xs_m = np.where(mask > 0)
        if ys_m.size:
            pad = max(8, round(0.02 * max(h, w)))
            y0 = max(0, int(ys_m.min()) - pad)
            y1 = min(h, int(ys_m.max()) + pad)
            x0 = max(0, int(xs_m.min()) - pad)
            x1 = min(w, int(xs_m.max()) + pad)
            label = read_label(get_ocr_reader(settings), bgr[y0:y1, x0:x1])

    # 평가·오토라벨링 산출물 (계획서 Step 1): 누끼 오버레이 · 마스크 · 폴리곤
    seg_assets_meta: dict[str, Any] = {}
    if settings.save_segmentation_assets:
        try:
            assets = build_assets(bgr, mask, card.quad_px if card else None)
            seg_assets_meta = assets.as_meta()
            prefix = f"segmentation/{job_id}"
            for name, payload, ctype in (
                ("overlay.jpg", assets.overlay_jpg, "image/jpeg"),
                ("mask.png", assets.mask_png, "image/png"),
                ("cutout.png", assets.cutout_png, "image/png"),
            ):
                upload_object(settings, f"{prefix}/{name}", payload, ctype)
            seg_assets_meta["assets"] = ["overlay.jpg", "mask.png", "cutout.png"]
        except Exception as e:  # noqa: BLE001
            # 산출물 저장 실패가 분석 실패가 되면 안 된다
            log.warning("segmentation assets failed: %s", e)
            seg_assets_meta = {"error": str(e)[:200]}

    V_adj, alpha_k, beta_k = hollow.adjusted_volume_depth_mm3(rec.volume_mm3, inp.product_k)
    mass = hollow.mass_g(V_adj, inp.metal, inp.purity)

    mass_cap_g = _sanity_mass_cap_g(inp.product_k, settings)
    implausible_mass = mass > mass_cap_g

    # 몸체 전체가 금이 아닌 제품(봉입·금박·도금)은 부피와 금 함량이 무관하다(§6.2).
    volume_unmeasurable = inp.product_k.lower() in VOLUME_UNMEASURABLE_PRODUCTS

    # 다만 이런 제품은 함유량이 **제품에 인쇄돼 있다**("순금 0.005g").
    # 표기값을 받으면 측정 대신 그걸 쓴다 — 우리 추정보다 훨씬 정확하다.
    # 측정값이 아니라는 사실은 mass_source 로 명시한다.
    declared_g = inp.declared_gold_g if (inp.declared_gold_g or 0) > 0 else None
    declared_from_ocr = False
    if declared_g is None and label.weight_g and label.weight_confidence >= settings.ocr_min_confidence:
        # 사용자가 안 넣었고 각인을 충분히 확신할 때만 쓴다.
        # 사용자 입력이 항상 우선 — 우리가 읽은 것보다 본인이 아는 게 낫다.
        declared_g = label.weight_g
        declared_from_ocr = True
    measured_mass = mass
    body_not_solid_gold = False
    declared_ratio: float | None = None

    if declared_g:
        # 표기값이 있으면 **측정 부피와 대조**해 몸체가 순금인지 판정할 수 있다.
        # 속이 꽉 찬 순금이라면 두 값이 비슷해야 한다. 측정이 표기보다 몇 배나
        # 크면 몸체 전체가 금은 아니라는 뜻 — 봉입형·금박·도금이다.
        # 실측: "FINE GOLD 0.05g" 각인 바를 goldbar 로 재면 1.7g 이 나온다.
        #       0.05g 을 그 면적에 펴면 2.9μm — 따로 제련한 순금 박을 봉입한 제품.
        declared_ratio = measured_mass / float(declared_g) if declared_g > 0 else None
        body_not_solid_gold = declared_ratio is not None and declared_ratio > SOLID_GOLD_RATIO_THRESHOLD
        mass = float(declared_g)
        volume_unmeasurable = False
        mass_source = "ocr_label" if declared_from_ocr else "declared_label"
    else:
        mass_source = "measured_volume"

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
        quality_ok=not implausible_mass and not volume_unmeasurable,
        precision_boost=K.is_reliable and scale_tight,
        coarse_volume_model=weak_model or thickness_assumed,
    )
    if mass_source in ("declared_label", "ocr_label"):
        # 표기값은 제조사 스펙이다 — 우리 측정 신뢰도로 깎을 대상이 아니다.
        # 다만 "우리가 잰 값"도 아니므로 medium 으로 두고 출처를 밝힌다.
        tier = "medium"
    else:
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

    vol_sigma = 0.0 if mass_source in ("declared_label", "ocr_label") else conf_mod.volume_relative_sigma(
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
    if inp.reference_thickness_mm:
        warnings.append(
            f"두께는 입력하신 {inp.reference_thickness_mm:g} mm 를 사용했습니다"
            "(이 두께는 사진으로 잴 수 없어 입력값에 정확도가 좌우됩니다)."
        )
    if thickness_assumed:
        warnings.append(
            f"두께를 관측하지 못해 제품 기준값({rec.h_mean_mm:.1f} mm)으로 가정했습니다. "
            "실제 무게와 차이가 클 수 있습니다."
        )
    if label.purity and label.purity != inp.purity.lower():
        warnings.append(
            f"각인은 «{label.purity_source_text}»({label.purity})로 보이는데 "
            f"선택하신 함량은 {inp.purity} 입니다. 함량이 다르면 무게도 달라집니다."
        )
    if not K.is_reliable:
        warnings.append("사진에 초점거리 정보(EXIF)가 없어 카메라 값을 추정했습니다.")
    if implausible_mass:
        warnings.append(
            f"추정 무게가 비현실적으로 큽니다(상한 {mass_cap_g:.0f} g 초과). 촬영을 다시 확인해 주세요."
        )

    if mass_source == "ocr_label":
        warnings.insert(
            0,
            f"제품 각인에서 «{label.weight_source_text}» 를 읽어 {declared_g:g} g 으로 "
            f"계산했습니다(인식 신뢰도 {label.weight_confidence:.0%}). "
            "다르면 함유량을 직접 입력해 주세요.",
        )
    elif mass_source == "declared_label":
        warnings.insert(
            0,
            f"입력하신 제품 표기 {declared_g:g} g 을 그대로 사용했습니다. "
            "아래 실측 치수는 제품 확인용 참고값입니다.",
        )
        if body_not_solid_gold and declared_ratio:
            warnings.insert(
                1,
                f"사진에서 잰 부피가 전부 순금이라면 {measured_mass:.2f} g 이어야 하는데 "
                f"표기는 {declared_g:g} g 입니다(약 {declared_ratio:.0f}배 차이). "
                "제품 몸체 전체가 순금은 아닌 것으로 보입니다 — 따로 제련한 순금 박·시트를 아크릴·수지에 봉입했거나 금박·도금인 경우입니다. "
                "금 자체는 표기대로여도 몸체 크기로는 함량을 알 수 없으므로 표기값을 쓰는 것이 맞습니다.",
            )
    if volume_unmeasurable:
        warnings.insert(
            0,
            "이 제품은 **부피로 금 함량을 알 수 없습니다.** 따로 제련한 순금 박·시트를 "
            "아크릴·수지에 봉입했거나 금박·도금이면 몸체 크기와 금 무게가 무관합니다. "
            "실제 함유량은 제품 표기(예: 순금 0.005g)를 따라 주세요. "
            "아래 실측 치수는 참고용입니다.",
        )

    sanity_meta: dict[str, Any] = {
        "suppress_mass_display": implausible_mass or volume_unmeasurable,
        "implausible_mass": implausible_mass,
        "volume_unmeasurable": volume_unmeasurable,
        "mass_source": mass_source,
        "body_not_solid_gold": body_not_solid_gold,
        "measured_mass_g": round(measured_mass, 4) if declared_g else None,
        "measured_over_declared_ratio": round(declared_ratio, 2) if declared_ratio else None,
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
            if implausible_mass or volume_unmeasurable
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
            "segmentation": {**mask_meta, **seg_assets_meta},
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
            "label_ocr": label.as_meta(),
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
        card = compute_card_geometry(bgr, view, settings, exif_meta[view])
        cards[view] = card
        sigmas[view] = card.sigma_mm_per_px
        mask, seg_meta = build_jewel_mask(bgr, card, settings, view, job_id=job_id)
        masks[view] = mask
        placement_by_view[view] = seg_meta.get("placement_mode", "subtract_card")
        bboxes[view] = jewel_bbox_uv_mm(mask, card, view)

    sigma_max_dev, sigma_outliers = _sigma_consistency(sigmas, settings.scale_mismatch_ratio)

    # 슬랩 교집합에서 손각대 오차로 붙인 축이 있으면 기록해 둔다
    relaxed_axes: list[str] = []
    voxel_mod.slab_aabb_intervals_mm(bboxes, relaxed_out=relaxed_axes)

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
    # σ 가 크게 흩어졌다 = 카드 검출이 흔들렸을 수 있다 → 거절 대신 신뢰도 감점
    scale_tight = len(fallback_views) == 0 and not sigma_outliers
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
    if sigma_outliers:
        sanity_meta["warnings"].append(
            f"{', '.join(sigma_outliers)} 컷의 촬영 거리가 서로 많이 달라 신뢰도를 낮췄습니다"
            f"(분석은 진행했습니다). 5장을 비슷한 거리에서 찍으면 정확도가 올라갑니다."
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
    # 손각대 오차로 슬랩 구간을 붙인 축이 있으면 그 사실을 남긴다(조용히 넘어가지 않음)
    degraded_reasons.extend(f"slab_relaxed:{r}" for r in relaxed_axes)
    degraded_reasons.extend(f"scale_spread:{v}" for v in sigma_outliers)

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
            "sigma_spread": {
                "max_relative_deviation": round(sigma_max_dev, 4),
                "warn_ratio": settings.scale_mismatch_ratio,
                "outlier_views": sigma_outliers,
            },
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
