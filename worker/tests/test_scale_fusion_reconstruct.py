"""
스케일 융합 + 역투영 회귀 (`archimedes-v2-single-photo.mdc` §3·§4).

합성 장면을 **정확한 투영 기하로 직접 생성**해서, 파이프라인이 알려진 실치수와
알려진 부피를 복원하는지 본다. 깊이 모델 없이도 전 구간을 검증할 수 있다.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.constants import ID1_HEIGHT_MM, ID1_WIDTH_MM
from app.pipeline.backends.types import DepthKind, DepthMap
from app.pipeline.camera import Intrinsics, intrinsics_from_exif
from app.pipeline.card import CardGeometry
from app.pipeline.exceptions import PipelineError
from app.pipeline.reconstruct import (
    SupportPlane,
    principal_extents_mm,
    projected_area_mm2,
    reconstruct_from_depth,
)
from app.pipeline.scale_fusion import (
    card_object_points_mm,
    fuse_scale,
    plane_depth_map,
    solve_card_plane,
)

IMG_W, IMG_H = 640, 480
CARD_Z_MM = 300.0  # 카드까지의 거리


def _K() -> Intrinsics:
    return Intrinsics(fx=800.0, fy=800.0, cx=IMG_W / 2.0, cy=IMG_H / 2.0, source="exif_35mm")


def _project(K: Intrinsics, xyz: np.ndarray) -> np.ndarray:
    """카메라 좌표계 점 → 픽셀."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return np.stack([K.fx * x / z + K.cx, K.fy * y / z + K.cy], axis=1)


def _fronto_card(K: Intrinsics, z_mm: float = CARD_Z_MM) -> CardGeometry:
    """카메라와 평행한(fronto-parallel) 카드 — 깊이가 상수라 s 가 식별되지 않는 케이스."""
    hx, hy = ID1_WIDTH_MM / 2.0, ID1_HEIGHT_MM / 2.0
    corners = np.array(
        [[-hx, -hy, z_mm], [hx, -hy, z_mm], [hx, hy, z_mm], [-hx, hy, z_mm]], dtype=np.float64
    )
    quad = _project(K, corners).astype(np.float32)
    return CardGeometry(sigma_mm_per_px=ID1_WIDTH_MM / float(quad[1, 0] - quad[0, 0]), quad_px=quad)


def _tilted_card(K: Intrinsics, tilt_deg: float = 35.0, z_mm: float = CARD_Z_MM) -> CardGeometry:
    """기울어진 카드 — 깊이가 변해 s 가 식별되는 케이스."""
    hx, hy = ID1_WIDTH_MM / 2.0, ID1_HEIGHT_MM / 2.0
    local = np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]])
    a = np.deg2rad(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    cam = local @ Rx.T + np.array([0.0, 0.0, z_mm])
    quad = _project(K, cam).astype(np.float32)
    long_px = float(np.linalg.norm(quad[1] - quad[0]))
    return CardGeometry(sigma_mm_per_px=ID1_WIDTH_MM / long_px, quad_px=quad)


def _plane_depth_image(card: CardGeometry, K: Intrinsics) -> np.ndarray:
    """카드 평면의 참 깊이를 이미지 전체에 채운 맵 (mm)."""
    n, d = solve_card_plane(card, K)
    ys, xs = np.mgrid[0:IMG_H, 0:IMG_W]
    rx = (xs - K.cx) / K.fx
    ry = (ys - K.cy) / K.fy
    return (d / (n[0] * rx + n[1] * ry + n[2])).astype(np.float32)


# ───────────────────────── 카메라 K ─────────────────────────


def test_intrinsics_from_35mm_equivalent() -> None:
    """f_px = f35 × 대각픽셀 / 43.267 — 폭만 쓰는 근사는 3:2 가 아닌 센서에서 어긋난다."""
    K = intrinsics_from_exif({"focal_length_35mm": 26.0}, 4032, 3024)
    diag = float(np.hypot(4032, 3024))
    assert K.fx == pytest.approx(26.0 * diag / np.hypot(36.0, 24.0), rel=1e-6)
    assert K.source == "exif_35mm"
    assert K.is_reliable


def test_intrinsics_fallback_is_flagged_unreliable() -> None:
    """폴백 K 는 초점거리를 추측한 것 → 신뢰도 감점 대상으로 표시돼야 한다."""
    K = intrinsics_from_exif({}, 1000, 800)
    assert K.source == "fallback"
    assert not K.is_reliable


# ───────────────────────── 카드 평면 PnP ─────────────────────────


def test_card_object_points_follow_long_edge() -> None:
    """카드가 세로로 누워도 85.60mm 축을 올바르게 잡아야 한다."""
    landscape = np.array([[0, 0], [200, 0], [200, 126], [0, 126]], dtype=np.float32)
    portrait = np.array([[0, 0], [126, 0], [126, 200], [0, 200]], dtype=np.float32)
    lo = card_object_points_mm(landscape)
    po = card_object_points_mm(portrait)
    assert abs(lo[1, 0] - lo[0, 0]) == pytest.approx(ID1_WIDTH_MM)
    assert abs(po[1, 0] - po[0, 0]) == pytest.approx(ID1_HEIGHT_MM)


def test_solve_card_plane_recovers_distance() -> None:
    """
    `d` 는 카메라 원점에서 평면까지의 **수직거리**다. 카드 중심이 z=300 이어도
    35도 기울면 d = 300·cos35° ≈ 245.7 이 정상 — 물리적으로 의미 있는 값은
    카드 중심 화소에서의 깊이이므로 그쪽으로 확인한다.
    """
    K = _K()
    for card, expected_d in (
        (_fronto_card(K), CARD_Z_MM),
        (_tilted_card(K), CARD_Z_MM * np.cos(np.deg2rad(35.0))),
    ):
        n, d = solve_card_plane(card, K)
        assert np.linalg.norm(n) == pytest.approx(1.0, rel=1e-6)
        assert d == pytest.approx(expected_d, rel=0.02)
        # 카드 중심 화소의 깊이는 두 경우 모두 300mm
        center = np.asarray(card.quad_px, dtype=np.float64).mean(axis=0)
        z_center = plane_depth_map(
            n, d, K, np.array([center[1]]), np.array([center[0]])
        )[0]
        assert z_center == pytest.approx(CARD_Z_MM, rel=0.02)


# ───────────────────────── 스케일 융합 ─────────────────────────


def test_fusion_recovers_absolute_scale_from_tilted_card() -> None:
    """깊이가 임의 스케일·오프셋으로 왜곡돼 있어도 앵커가 mm 로 되돌린다."""
    K = _K()
    card = _tilted_card(K)
    truth = _plane_depth_image(card, K)
    warped = DepthMap(depth=(truth * 0.004 + 7.0).astype(np.float32), kind=DepthKind.RELATIVE)

    res = fuse_scale(warped, K, card)
    assert res.method == "anchor_affine"
    assert res.anchor_used and not res.ill_conditioned
    assert res.scale_s == pytest.approx(1 / 0.004, rel=1e-3)
    assert np.allclose(res.depth_mm, truth, rtol=1e-3, atol=0.5)
    assert res.depth_rmse_mm is not None
    assert res.depth_rmse_mm < 0.5  # §7.1 홀드아웃 거리 RMSE


def test_constant_depth_is_ill_conditioned_but_still_anchored() -> None:
    """
    상수 깊이(스텁)에서는 s 가 식별되지 않는다. 억지 회귀로 s 를 폭주시키지 않고
    오프셋만 맞춘 뒤 ill_conditioned 로 알려야 한다.
    """
    K = _K()
    card = _fronto_card(K)
    flat = DepthMap(
        depth=np.ones((IMG_H, IMG_W), dtype=np.float32), kind=DepthKind.AFFINE_INVARIANT
    )
    res = fuse_scale(flat, K, card)
    assert res.ill_conditioned
    assert res.method == "anchor_offset_only"
    assert res.scale_s == pytest.approx(1.0)
    # 카드 평면 거리로 맞춰진다 → v1 약원근 가정과 동일한 퇴화 동작
    assert float(res.depth_mm.mean()) == pytest.approx(CARD_Z_MM, rel=0.02)


def test_no_anchor_with_unscaled_depth_raises() -> None:
    K = _K()
    rel = DepthMap(depth=np.ones((IMG_H, IMG_W), dtype=np.float32), kind=DepthKind.RELATIVE)
    with pytest.raises(PipelineError) as ei:
        fuse_scale(rel, K, None)
    assert ei.value.code == "ERR_SCALE_UNRESOLVED"


def test_no_anchor_with_metric_depth_passes_through() -> None:
    K = _K()
    metric = DepthMap(
        depth=np.full((IMG_H, IMG_W), 250.0, dtype=np.float32), kind=DepthKind.METRIC
    )
    res = fuse_scale(metric, K, None)
    assert res.method == "metric_passthrough"
    assert not res.anchor_used
    assert res.depth_mm.mean() == pytest.approx(250.0)


def test_fabricated_card_quad_is_not_used_as_anchor() -> None:
    """폴백 쿼드는 우리가 지어낸 사각형이라 앵커로 쓰면 스케일이 날조된다."""
    K = _K()
    card = _fronto_card(K)
    card.used_fallback_quad = True
    rel = DepthMap(depth=np.ones((IMG_H, IMG_W), dtype=np.float32), kind=DepthKind.RELATIVE)
    with pytest.raises(PipelineError) as ei:
        fuse_scale(rel, K, card)
    assert ei.value.code == "ERR_SCALE_UNRESOLVED"


def test_require_anchor_rejects_metric_without_card() -> None:
    K = _K()
    metric = DepthMap(depth=np.full((IMG_H, IMG_W), 250.0, np.float32), kind=DepthKind.METRIC)
    with pytest.raises(PipelineError) as ei:
        fuse_scale(metric, K, None, require_anchor=True)
    assert ei.value.code == "ERR_SCALE_UNRESOLVED"


# ───────────────────────── 역투영·부피 ─────────────────────────


def test_projected_area_matches_known_square() -> None:
    """한 변 20mm 정사각형을 300mm 거리에 두면 면적 400mm² 가 복원돼야 한다."""
    K = _K()
    side_mm = 20.0
    side_px = side_mm * K.fx / CARD_Z_MM
    mask = np.zeros((IMG_H, IMG_W), np.uint8)
    x0 = int(K.cx - side_px / 2)
    y0 = int(K.cy - side_px / 2)
    cv2.rectangle(mask, (x0, y0), (x0 + int(side_px) - 1, y0 + int(side_px) - 1), 255, -1)
    depth = np.full((IMG_H, IMG_W), CARD_Z_MM, np.float32)

    z = depth[mask > 0].astype(np.float64)
    assert projected_area_mm2(z, K) == pytest.approx(side_mm**2, rel=0.05)


def test_area_ignores_ring_hole_unlike_bbox() -> None:
    """
    v1 대비 핵심 개선: 반지처럼 가운데가 빈 형상에서 bbox 는 구멍까지 세지만
    마스크 실면적은 세지 않는다.
    """
    K = _K()
    depth = np.full((IMG_H, IMG_W), CARD_Z_MM, np.float32)
    c = (int(K.cx), int(K.cy))
    annulus = np.zeros((IMG_H, IMG_W), np.uint8)
    cv2.circle(annulus, c, 60, 255, -1)
    cv2.circle(annulus, c, 45, 0, -1)
    disc = np.zeros((IMG_H, IMG_W), np.uint8)
    cv2.circle(disc, c, 60, 255, -1)

    a_ring = projected_area_mm2(depth[annulus > 0].astype(np.float64), K)
    a_disc = projected_area_mm2(depth[disc > 0].astype(np.float64), K)
    assert a_ring < a_disc * 0.5


def test_principal_extents_of_known_slab() -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform(low=[-20, -8, -1.5], high=[20, 8, 1.5], size=(20000, 3))
    length, width, h = principal_extents_mm(pts)
    assert length == pytest.approx(40.0, rel=0.1)
    assert width == pytest.approx(16.0, rel=0.1)
    assert h == pytest.approx(3.0, rel=0.15)


def _flat_table(z_mm: float = CARD_Z_MM) -> SupportPlane:
    return SupportPlane(normal=np.array([0.0, 0.0, 1.0]), d_mm=z_mm)


def _square_mask(K: Intrinsics, side_mm: float, at_z: float) -> np.ndarray:
    side_px = side_mm * K.fx / at_z
    mask = np.zeros((IMG_H, IMG_W), np.uint8)
    x0, y0 = int(K.cx - side_px / 2), int(K.cy - side_px / 2)
    cv2.rectangle(mask, (x0, y0), (x0 + int(side_px) - 1, y0 + int(side_px) - 1), 255, -1)
    return mask


def test_height_field_recovers_known_plate_volume() -> None:
    """
    20×20mm, 두께 3mm 판이 바닥에 놓여 있음 → 1200 mm³.

    핵심: 한 장의 사진으로는 두께를 볼 수 없지만, **물체가 카드와 같은 바닥면에
    있다**는 프로토콜을 쓰면 바닥까지의 높이로 부피가 결정된다.
    """
    K = _K()
    side_mm, thickness = 20.0, 3.0
    top_z = CARD_Z_MM - thickness
    mask = _square_mask(K, side_mm, top_z)
    depth = np.full((IMG_H, IMG_W), top_z, np.float32)

    rec = reconstruct_from_depth(mask, depth, K, "ring", support_plane=_flat_table())
    assert rec.method == "height_field"
    assert rec.area_proj_mm2 == pytest.approx(side_mm**2, rel=0.06)
    assert rec.h_mean_mm == pytest.approx(thickness, rel=0.05)
    assert rec.thickness_clamp is None
    assert rec.volume_mm3 == pytest.approx(side_mm**2 * thickness, rel=0.08)


def test_height_field_ignores_surface_behind_the_table() -> None:
    """바닥보다 뒤에 있는 표본(그림자·배경 누수)은 음수 부피로 새면 안 된다."""
    K = _K()
    mask = _square_mask(K, 20.0, CARD_Z_MM)
    depth = np.full((IMG_H, IMG_W), CARD_Z_MM + 25.0, np.float32)  # 전부 바닥 뒤
    rec = reconstruct_from_depth(mask, depth, K, "ring", support_plane=_flat_table())
    assert rec.volume_mm3 > 0  # 최소 두께로 클램프
    assert rec.thickness_clamp == "min"


def test_flat_on_table_clamps_to_min_thickness() -> None:
    """표면이 바닥과 같은 높이면 부피 0 → 제품별 최소 두께로 클램프 + 플래그."""
    K = _K()
    mask = _square_mask(K, 20.0, CARD_Z_MM)
    depth = np.full((IMG_H, IMG_W), CARD_Z_MM, np.float32)
    rec = reconstruct_from_depth(mask, depth, K, "ring", support_plane=_flat_table())
    assert rec.thickness_clamp == "min"
    assert rec.h_mean_mm == pytest.approx(1.0, rel=1e-3)


def test_thickness_clamped_at_max() -> None:
    K = _K()
    top_z = CARD_Z_MM - 40.0  # 물리적으로 말 안 되는 두께
    mask = _square_mask(K, 20.0, top_z)
    depth = np.full((IMG_H, IMG_W), top_z, np.float32)
    rec = reconstruct_from_depth(mask, depth, K, "ring", support_plane=_flat_table())
    assert rec.thickness_clamp == "max"
    assert rec.h_mean_mm == pytest.approx(9.0, rel=1e-3)


def test_prism_fallback_is_weak_on_tilted_plane() -> None:
    """
    앵커가 없으면 PCA 두께로 폴백하는데, 기울어진 평평한 판은 3주축 치수가
    0 에 가깝다 — 원리적 한계라 최소 두께로 떨어진다. 이 모드가 약하다는 것을
    테스트로 못박아 둔다(그래서 앵커가 기본이다).
    """
    K = _K()
    mask = _square_mask(K, 20.0, CARD_Z_MM)
    ys = np.broadcast_to(np.arange(IMG_H)[:, None], (IMG_H, IMG_W)).astype(np.float32)
    depth = (CARD_Z_MM + (ys - K.cy) * 0.05).astype(np.float32)  # 기울어진 평면

    rec = reconstruct_from_depth(mask, depth, K, "ring")
    assert rec.method == "prism_pca"
    assert rec.h_vis_mm < 0.2
    assert rec.thickness_clamp == "min"


def test_reconstruct_requires_enough_points() -> None:
    K = _K()
    mask = np.zeros((IMG_H, IMG_W), np.uint8)
    mask[0:2, 0:2] = 255
    depth = np.full((IMG_H, IMG_W), CARD_Z_MM, np.float32)
    with pytest.raises(PipelineError) as ei:
        reconstruct_from_depth(mask, depth, K, "ring")
    assert ei.value.code == "ERR_DEPTH_FAILED"


def test_volume_scales_cubically_with_distance_error() -> None:
    """
    §3 의 근거: 거리 10% 오차 → 부피 33% 오차.
    스케일 융합이 왜 필수인지 수치로 고정해 둔다.
    """
    K = _K()
    thickness_ratio = 3.0 / CARD_Z_MM
    mask = _square_mask(K, 20.0, CARD_Z_MM - 3.0)

    def vol(table_z: float) -> float:
        top = table_z * (1.0 - thickness_ratio)
        depth = np.full((IMG_H, IMG_W), top, np.float32)
        return reconstruct_from_depth(
            mask, depth, K, "ring", support_plane=_flat_table(table_z)
        ).volume_mm3

    assert vol(CARD_Z_MM * 1.1) / vol(CARD_Z_MM) == pytest.approx(1.1**3, rel=0.02)
