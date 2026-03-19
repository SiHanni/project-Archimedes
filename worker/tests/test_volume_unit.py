from app.pipeline.voxel import volume_from_view_bboxes


def test_volume_intersection_box() -> None:
    # 10mm cube in consistent slab constraints
    b = {
        "front": (-5, 5, -5, 5),  # x, z
        "top": (-5, 5, -5, 5),  # x, y
        "left": (-5, 5, -5, 5),  # z, y
        "right": (-5, 5, -5, 5),
        "back": (-5, 5, -5, 5),
    }
    v = volume_from_view_bboxes(b)
    assert abs(v - 1000.0) < 1e-3
