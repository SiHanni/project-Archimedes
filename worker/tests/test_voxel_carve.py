"""복셀 카빙 경로 스모크 (격자 작게)."""

from __future__ import annotations

import numpy as np

from app.pipeline.card import CardGeometry
from app.pipeline.voxel import (
    carve_visual_hull_mm3,
    slab_aabb_intervals_mm,
    volume_from_view_bboxes,
)


def _fake_card(h: int = 512, w: int = 512) -> CardGeometry:
    quad = np.array(
        [[80.0, 60.0], [420.0, 55.0], [425.0, 310.0], [75.0, 305.0]],
        dtype=np.float32,
    )
    return CardGeometry(
        sigma_mm_per_px=0.1,
        quad_px=quad,
        warped_preview=None,
        homography_3x3=np.eye(3, dtype=np.float64),
        precision_pose_candidate=False,
        precision_solution_count=0,
    )


def test_carve_volume_not_exceeds_slab() -> None:
    """전부 흰 마스크면 슬랩 상자 전체가 살아 카빙 부피 ≈ 슬랩(근사)."""
    bboxes = {
        "front": (-50.0, 50.0, -40.0, 40.0),
        "top": (-50.0, 50.0, -30.0, 30.0),
        "left": (-40.0, 40.0, -30.0, 30.0),
        "right": (-40.0, 40.0, -30.0, 30.0),
        "back": (-50.0, 50.0, -40.0, 40.0),
    }
    card = _fake_card()
    mask = np.full((512, 512), 255, dtype=np.uint8)
    views = [(v, mask, card) for v in ("front", "top", "left", "right", "back")]
    v_sl = volume_from_view_bboxes(bboxes)
    v_c = carve_visual_hull_mm3(views, bboxes, grid_n=24)
    assert v_c > 0
    # 카빙 AABB는 슬랩 교차에 3% 팽창이 붙어 슬랩 단독 부피보다 다소 클 수 있음
    assert v_c <= v_sl * 1.25 + 5000.0
    x_rng, _y_rng, _z_rng = slab_aabb_intervals_mm(bboxes)
    assert x_rng[0] < x_rng[1]
