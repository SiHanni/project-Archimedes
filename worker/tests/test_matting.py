"""
누끼 정밀화 테스트.

외형 폴백의 Otsu 는 금의 반사로 빛나는 **일부만** 잡는다(실측: 금괴 상단 40%).
GrabCut 이 색 분포로 나머지를 끌어와야 하고, 동시에 폭주하면 안 된다.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.matting import refine_with_grabcut


def _scene_with_partial_seed() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """어두운 바닥 위 균일한 금색 막대 + 그 위쪽 절반만 덮은 씨앗."""
    rng = np.random.default_rng(7)
    img = np.full((400, 600, 3), 35, np.uint8)
    img += rng.integers(0, 8, img.shape, dtype=np.int16).astype(np.uint8)

    x0, y0, x1, y1 = 200, 150, 420, 260
    img[y0:y1, x0:x1] = (60, 190, 235)  # BGR 금색
    img = cv2.GaussianBlur(img, (3, 3), 0)

    seed = np.zeros(img.shape[:2], np.uint8)
    seed[y0 : y0 + (y1 - y0) // 2, x0:x1] = 255  # 위쪽 절반만

    truth = np.zeros(img.shape[:2], np.uint8)
    truth[y0:y1, x0:x1] = 255
    return img, seed, truth


def test_grabcut_expands_partial_seed_to_whole_object():
    img, seed, truth = _scene_with_partial_seed()
    out, meta = refine_with_grabcut(img, seed)

    assert meta["matting"] == "grabcut"
    seed_area = int(cv2.countNonZero(seed))
    out_area = int(cv2.countNonZero(out))
    truth_area = int(cv2.countNonZero(truth))

    # 씨앗보다 확실히 커지고, 정답 면적에 근접해야 한다
    assert out_area > seed_area * 1.4
    assert abs(out_area - truth_area) / truth_area < 0.20

    inter = int(cv2.countNonZero(cv2.bitwise_and(out, truth)))
    union = int(cv2.countNonZero(cv2.bitwise_or(out, truth)))
    assert inter / union > 0.80, f"IoU={inter / union:.3f}"


def test_grabcut_keeps_out_of_excluded_region():
    """카드 영역을 확정 배경으로 주면 그쪽으로 새지 않는다."""
    img, seed, _truth = _scene_with_partial_seed()
    # 막대 바로 오른쪽에 같은 색 '카드'를 붙여 둔다
    img[150:260, 425:560] = (60, 190, 235)
    exclude = np.zeros(img.shape[:2], np.uint8)
    exclude[150:260, 425:560] = 255

    out, _meta = refine_with_grabcut(img, seed, exclude=exclude)
    leaked = int(cv2.countNonZero(cv2.bitwise_and(out, exclude)))
    assert leaked < int(cv2.countNonZero(exclude)) * 0.05


def test_grabcut_returns_seed_when_it_would_explode():
    """배경과 구분이 안 되는 장면에서는 원본 마스크를 그대로 돌려준다."""
    rng = np.random.default_rng(3)
    img = rng.integers(90, 110, (300, 300, 3), dtype=np.uint8)  # 균일 노이즈뿐
    seed = np.zeros((300, 300), np.uint8)
    seed[140:160, 140:160] = 255

    out, meta = refine_with_grabcut(img, seed)
    if meta["matting"] != "grabcut":
        assert np.array_equal(out, seed)
    else:
        # 통과했다면 최소한 폭주 한도 안이어야 한다
        assert float(meta["grabcut_growth"]) <= 8.0


def test_grabcut_skips_tiny_seed():
    img = np.zeros((100, 100, 3), np.uint8)
    seed = np.zeros((100, 100), np.uint8)
    seed[50:53, 50:53] = 255
    out, meta = refine_with_grabcut(img, seed)
    assert meta["matting"] == "skipped_small_seed"
    assert np.array_equal(out, seed)
