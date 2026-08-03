"""
기하 코어 회귀 테스트 — **알려진 치수 → 알려진 부피**.

이 파일이 없어서 `archimedes-v2-single-photo.mdc` §0.4 의 결함 #1~#4 가
장기간 잡히지 않았다. 각 테스트는 특정 결함에 1:1 대응한다.

핵심 아이디어: 월드 AABB 를 `world_mm_to_pixel_uv` 로 각 뷰에 투영해 마스크를 만들고,
파이프라인이 그 마스크로부터 **원래 AABB 를 복원**하는지 본다(round-trip).
약원근 투영은 아핀이라 박스의 실루엣은 정확히 직사각형이므로 기대값이 해석적으로 정해진다.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.constants import ID1_HEIGHT_MM, ID1_WIDTH_MM, VIEW_ORDER
from app.pipeline.card import CardGeometry, sigma_mm_per_px_from_quad
from app.pipeline.exceptions import PipelineError
from app.pipeline.geometry_g1 import jewel_bbox_uv_mm
from app.pipeline.geometry_project import sample_mask, world_mm_to_pixel_uv
from app.pipeline.ingest import apply_exif_orientation
from app.pipeline.view_axes import view_world_intervals
from app.pipeline.voxel import (
    carve_visual_hull_mm3,
    slab_aabb_intervals_mm,
    volume_from_view_bboxes,
)

IMG_W, IMG_H = 1600, 1200
SIGMA = 0.1  # mm/px

# 카드 중심에서 **비대칭으로 떨어진** 박스 — 축·부호 오류를 잡으려면 오프셋이 필수다.
# (원점 대칭 박스는 좌/우 축이 뒤바뀌어도 통과해 버린다)
BOX = {"x": (10.0, 26.0), "y": (-30.0, -14.0), "z": (0.0, 8.0)}


def _card_with_sigma(sigma: float, img_w: int = IMG_W, img_h: int = IMG_H) -> CardGeometry:
    """원하는 σ 가 나오도록 ID-1 비율 쿼드를 만든다."""
    long_px = ID1_WIDTH_MM / sigma
    short_px = long_px * (ID1_HEIGHT_MM / ID1_WIDTH_MM)
    cx, cy = img_w / 2.0, img_h / 2.0
    quad = np.array(
        [
            [cx - long_px / 2, cy - short_px / 2],
            [cx + long_px / 2, cy - short_px / 2],
            [cx + long_px / 2, cy + short_px / 2],
            [cx - long_px / 2, cy + short_px / 2],
        ],
        dtype=np.float32,
    )
    return CardGeometry(sigma_mm_per_px=sigma_mm_per_px_from_quad(quad), quad_px=quad)


def _mask_of_box(view: str, card: CardGeometry, box: dict) -> np.ndarray:
    """월드 AABB 를 해당 뷰에 투영한 실루엣(직사각형) 마스크."""
    xs, ys, zs = [], [], []
    for x, y, z in itertools.product(box["x"], box["y"], box["z"]):
        xs.append(x)
        ys.append(y)
        zs.append(z)
    px, py = world_mm_to_pixel_uv(
        view, np.array(xs), np.array(ys), np.array(zs), card
    )
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    x0, x1 = int(np.floor(px.min())), int(np.ceil(px.max()))
    y0, y1 = int(np.floor(py.min())), int(np.ceil(py.max()))
    assert 0 <= x0 < x1 < IMG_W and 0 <= y0 < y1 < IMG_H, f"{view}: box projects out of frame"
    mask[y0 : y1 + 1, x0 : x1 + 1] = 255
    return mask


def _bboxes_for_box(card: CardGeometry, box: dict) -> dict[str, tuple[float, float, float, float]]:
    return {v: jewel_bbox_uv_mm(_mask_of_box(v, card, box), card, v) for v in VIEW_ORDER}


# ─────────────────────────── #1 σ 스케일 ───────────────────────────


def test_sigma_scales_with_card_pixel_size() -> None:
    """
    결함 #1 회귀: σ 는 **원본 이미지에서의 카드 크기**에 반비례해야 한다.

    이전 구현은 워프 캔버스 폭(856)으로 나눠 σ 가 카드 크기와 무관하게 항상 0.1 이었다.
    """
    small = _card_with_sigma(0.2)  # 카드가 작게 찍힘 → mm/px 큼
    large = _card_with_sigma(0.05)  # 카드가 크게 찍힘 → mm/px 작음
    assert small.sigma_mm_per_px == pytest.approx(0.2, rel=1e-6)
    assert large.sigma_mm_per_px == pytest.approx(0.05, rel=1e-6)
    assert small.sigma_mm_per_px > large.sigma_mm_per_px * 3


def test_sigma_independent_of_quad_rotation_order() -> None:
    """카드가 세로로 누워도(긴 변이 TL→TR 이 아니어도) σ 는 같아야 한다."""
    q = _card_with_sigma(0.1).quad_px
    rotated = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)  # 코너 1칸 회전
    assert sigma_mm_per_px_from_quad(rotated) == pytest.approx(0.1, rel=1e-6)


def test_degenerate_quad_rejected() -> None:
    tiny = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    with pytest.raises(PipelineError) as ei:
        sigma_mm_per_px_from_quad(tiny)
    assert ei.value.code == "ERR_CARD_NOT_FOUND"


# ─────────────────────── #2 뷰↔월드 축 매핑 ───────────────────────


def test_side_views_map_image_vertical_to_height() -> None:
    """
    결함 #2 회귀: 옆에서 본 뷰는 **이미지 세로 = 높이(z)**, 가로 = 깊이(y).

    이전 구현은 좌/우를 `u~z, v~y` 로 뒤집어 매핑했다.
    """
    for view in ("left", "right"):
        axes = view_world_intervals(view, (-5.0, 5.0, 1.0, 9.0))
        assert set(axes) == {"y", "z"}, view
        assert axes["z"] == (1.0, 9.0), f"{view}: v(이미지 세로)가 z 로 가야 한다"


def test_front_back_and_top_axis_coverage() -> None:
    assert set(view_world_intervals("front", (0, 1, 0, 1))) == {"x", "z"}
    assert set(view_world_intervals("back", (0, 1, 0, 1))) == {"x", "z"}
    assert set(view_world_intervals("top", (0, 1, 0, 1))) == {"x", "y"}


def test_projection_roundtrip_recovers_offset_box() -> None:
    """
    5뷰 실루엣 → 슬랩 AABB 가 **원래 박스를 복원**해야 한다.

    박스가 카드 중심에서 비대칭으로 떨어져 있으므로, 축 스왑이나 부호 오류가 있으면
    교집합이 비거나(ERR_VOLUME) 엉뚱한 구간이 나온다.
    """
    card = _card_with_sigma(SIGMA)
    bboxes = _bboxes_for_box(card, BOX)
    x_rng, y_rng, z_rng = slab_aabb_intervals_mm(bboxes)

    tol = 3 * SIGMA  # 마스크 픽셀 양자화 여유
    assert x_rng[0] == pytest.approx(BOX["x"][0], abs=tol)
    assert x_rng[1] == pytest.approx(BOX["x"][1], abs=tol)
    assert y_rng[0] == pytest.approx(BOX["y"][0], abs=tol)
    assert y_rng[1] == pytest.approx(BOX["y"][1], abs=tol)
    assert z_rng[0] == pytest.approx(BOX["z"][0], abs=tol)
    assert z_rng[1] == pytest.approx(BOX["z"][1], abs=tol)


def test_known_dimensions_give_known_volume() -> None:
    """16 × 16 × 8 mm = 2048 mm³ 를 슬랩·카빙 양쪽에서 복원."""
    card = _card_with_sigma(SIGMA)
    bboxes = _bboxes_for_box(card, BOX)
    expected = 16.0 * 16.0 * 8.0

    v_slab = volume_from_view_bboxes(bboxes)
    assert v_slab == pytest.approx(expected, rel=0.06)

    view_items = [(v, _mask_of_box(v, card, BOX), card) for v in VIEW_ORDER]
    v_carve = carve_visual_hull_mm3(view_items, bboxes, grid_n=48)
    # 카빙 격자는 3% 팽창된 AABB 위에 놓이므로 이산화 오차를 넉넉히 본다
    assert v_carve == pytest.approx(expected, rel=0.15)
    assert v_carve <= v_slab * 1.05


def test_scale_change_propagates_cubically_to_volume() -> None:
    """σ 가 2배면 같은 마스크의 부피는 8배 — 스케일이 실제로 부피를 지배하는지 확인."""
    card_a = _card_with_sigma(SIGMA)
    card_b = _card_with_sigma(SIGMA * 2)
    box_b = {k: (v[0] * 2, v[1] * 2) for k, v in BOX.items()}
    v_a = volume_from_view_bboxes(_bboxes_for_box(card_a, BOX))
    v_b = volume_from_view_bboxes(_bboxes_for_box(card_b, box_b))
    assert v_b / v_a == pytest.approx(8.0, rel=0.08)


# ───────────────────── #3 모순된 뷰 → ERR_VOLUME ─────────────────────


def test_contradicting_views_raise_err_volume() -> None:
    """
    결함 #3 회귀: 교집합이 비면 **에러**여야 한다.

    이전 구현은 빈 교집합에 None 을 돌려주고, 다음 뷰가 그것을 '제약 없음'으로
    오인해 자기 구간으로 덮어써서 모순이 조용히 사라졌다.
    """
    bboxes = {
        "front": (-5.0, 5.0, 0.0, 8.0),  # x ∈ (-5,5)
        "top": (100.0, 110.0, -4.0, 4.0),  # x ∈ (100,110)  ← front 와 모순
        "left": (0.0, 8.0, -4.0, 4.0),
        "right": (0.0, 8.0, -4.0, 4.0),
        "back": (-30.0, -20.0, 0.0, 8.0),
    }
    with pytest.raises(PipelineError) as ei:
        slab_aabb_intervals_mm(bboxes)
    assert ei.value.code == "ERR_VOLUME"
    assert ei.value.retry_step in ("top", "back")


def test_missing_axis_constraint_raises() -> None:
    with pytest.raises(PipelineError) as ei:
        slab_aabb_intervals_mm({"front": (-5.0, 5.0, 0.0, 8.0)})  # y 를 제약하는 뷰 없음
    assert ei.value.code == "ERR_VOLUME"


# ───────────────────── #4 EXIF orientation ─────────────────────


def test_exif_orientation_rotates_pixels() -> None:
    """
    결함 #4 회귀: orientation 6/8 은 가로·세로를 바꿔야 한다.

    적용하지 않으면 세로로 찍은 폰 사진이 눕혀진 채 들어가 축 매핑이 통째로 깨진다.
    """
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[0, 0] = 255  # 좌상단 마커

    assert apply_exif_orientation(img, 1).shape == (10, 20, 3)
    assert apply_exif_orientation(img, None).shape == (10, 20, 3)
    for o in (5, 6, 7, 8):
        assert apply_exif_orientation(img, o).shape == (20, 10, 3), f"orientation {o}"
    for o in (2, 3, 4):
        assert apply_exif_orientation(img, o).shape == (10, 20, 3), f"orientation {o}"

    # orientation 6 = 시계 90도 → 좌상단 마커가 우상단으로
    rot = apply_exif_orientation(img, 6)
    assert rot[0, -1, 0] == 255


# ───────────────── sample_mask 프레임 밖 처리 ─────────────────


def test_sample_mask_treats_out_of_frame_as_outside() -> None:
    """이전 구현은 clip 으로 테두리에 붙여, 테두리가 전경이면 화면 밖도 내부로 셌다."""
    mask = np.full((10, 10), 255, dtype=np.uint8)
    px = np.array([5.0, -3.0, 12.0, 5.0])
    py = np.array([5.0, 5.0, 5.0, 99.0])
    got = sample_mask(mask, px, py)
    assert got.tolist() == [True, False, False, False]


# ───────────────── 손각대 허용 오차 ─────────────────


def test_small_axis_mismatch_is_relaxed_not_rejected() -> None:
    """
    손각대 5뷰는 축 정렬이 완벽할 수 없다. 살짝 어긋난 정도까지 ERR_VOLUME 으로
    떨어뜨리면 정상 촬영도 전부 거절된다. 붙여 주되 사유를 남긴다.
    """
    bboxes = {
        "front": (-5.0, 5.0, 0.0, 8.0),
        "top": (5.6, 15.0, -4.0, 4.0),  # x 가 front 와 0.6mm 만 벌어짐
        "left": (0.0, 8.0, -4.0, 4.0),
        "right": (0.0, 8.0, -4.0, 4.0),
        "back": (-5.0, 5.0, 0.0, 8.0),
    }
    relaxed: list[str] = []
    x_rng, _y, _z = slab_aabb_intervals_mm(bboxes, relaxed_out=relaxed)
    assert relaxed, "완화했으면 사유가 남아야 한다"
    assert x_rng[0] < x_rng[1]


def test_large_axis_mismatch_still_raises() -> None:
    """완화는 손각대 오차용이지 '다른 물체를 찍은' 경우까지 덮으면 안 된다."""
    bboxes = {
        "front": (-5.0, 5.0, 0.0, 8.0),
        "top": (100.0, 110.0, -4.0, 4.0),
        "left": (0.0, 8.0, -4.0, 4.0),
        "right": (0.0, 8.0, -4.0, 4.0),
        "back": (-5.0, 5.0, 0.0, 8.0),
    }
    with pytest.raises(PipelineError) as ei:
        slab_aabb_intervals_mm(bboxes)
    assert ei.value.code == "ERR_VOLUME"
