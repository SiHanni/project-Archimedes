"""
누끼 정밀화 테스트.

외형 폴백의 Otsu 는 금의 반사로 빛나는 **일부만** 잡는다(실측: 금괴 상단 40%).
GrabCut 이 색 분포로 나머지를 끌어와야 하고, 동시에 폭주하면 안 된다.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

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


def test_border_touching_component_is_rejected():
    """
    프레임 가장자리에 닿는 성분은 배경이다.

    실측(반지 사진): ROI 원이 프레임 위쪽을 넘어서 컵·케이블이 후보가 됐고,
    그게 최대 성분이라 반지 대신 채택됐다(15.604 g).
    """
    from app.pipeline.jewel_mask import touches_frame_border

    shape = (4032, 3024)
    # 위쪽 가장자리에 붙은 덩어리
    edge = np.zeros(5, np.int32)
    edge[cv2.CC_STAT_LEFT], edge[cv2.CC_STAT_TOP] = 800, 0
    edge[cv2.CC_STAT_WIDTH], edge[cv2.CC_STAT_HEIGHT] = 600, 300
    assert touches_frame_border(edge, shape)

    # 프레임 안쪽에 통째로 있는 덩어리
    inner = np.zeros(5, np.int32)
    inner[cv2.CC_STAT_LEFT], inner[cv2.CC_STAT_TOP] = 600, 2300
    inner[cv2.CC_STAT_WIDTH], inner[cv2.CC_STAT_HEIGHT] = 700, 700
    assert not touches_frame_border(inner, shape)


def test_chroma_finds_metal_that_matches_background_brightness():
    """
    명도가 배경과 같아도 채도로 금속을 찾는다.

    실측(반지 사진): 밝은 베이지 책상 위 금반지에서 Otsu 는 반지 안쪽 구멍의
    **그림자**를 물체로 잡았다.
    """
    from app.pipeline.appearance import chroma_foreground, local_lab_contrast

    img = np.full((400, 400, 3), 170, np.uint8)  # 밝은 무채색 책상
    cv2.circle(img, (200, 200), 90, (60, 170, 215), 18)  # 금색 고리
    cv2.circle(img, (200, 200), 72, (150, 150, 150), -1)  # 고리 안쪽 그림자

    fg = chroma_foreground(img)
    ring_hit = int(cv2.countNonZero(fg[110:130, 190:210]))  # 고리 위쪽 밴드
    hole = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(hole, (200, 200), 60, 255, -1)
    hole_hit = int(cv2.countNonZero(cv2.bitwise_and(fg, hole)))

    assert ring_hit > 0, "금속 밴드를 못 잡았다"
    assert hole_hit < int(cv2.countNonZero(hole)) * 0.2, "안쪽 그림자를 물체로 잡았다"
    assert local_lab_contrast(img, fg) > 5.0


def test_outline_finds_object_and_claims_no_scale():
    """
    에라토스테네스 경로: 기준물 없이 외곽선만 낸다.

    크기를 **주장하지 않는다**는 것이 이 경로의 계약이다. 소비 측이 없는 값을
    0 으로 채워 읽지 않도록 `scale_available: False` 를 명시한다.
    """
    from app.pipeline.eratosthenes import extract_outline

    rng = np.random.default_rng(11)
    img = np.full((600, 800, 3), 40, np.uint8)
    img += rng.integers(0, 6, img.shape, dtype=np.int16).astype(np.uint8)
    # 화면 가운데 금색 물체 하나
    cv2.ellipse(img, (400, 300), (90, 55), 20, 0, 360, (60, 185, 230), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    res = extract_outline(img)
    assert res.meta["scale_available"] is False
    ys, xs = np.where(res.mask > 0)
    assert 250 < xs.mean() < 550 and 200 < ys.mean() < 400, "물체 위치를 못 찾았다"
    frac = int(cv2.countNonZero(res.mask)) / (600 * 800)
    assert 0.005 < frac < 0.10, f"마스크 면적이 이상하다 {frac}"


def test_outline_ignores_border_touching_background():
    """화면 밖으로 이어지는 밝은 영역(배경)은 물체가 아니다."""
    from app.pipeline.eratosthenes import extract_outline

    img = np.full((600, 800, 3), 40, np.uint8)
    # 위쪽 가장자리에 붙은 큰 밝은 덩어리 = 배경
    img[0:150, 100:700] = (200, 200, 200)
    # 가운데 작은 물체
    cv2.circle(img, (400, 380), 60, (60, 185, 230), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    res = extract_outline(img)
    ys, _xs = np.where(res.mask > 0)
    assert ys.mean() > 250, "가장자리에 붙은 배경을 물체로 잡았다"


def test_false_card_detection_does_not_erase_the_object():
    """
    카드가 없는 사진에서 오검출된 '카드'가 물체를 지우면 안 된다.

    실측(도련님 사진): 카드가 없는데 검출기가 화면의 24.4% 를 카드로 잡았고,
    그 제외 영역이 금괴를 통째로 덮어 "귀금속을 찾지 못했습니다"가 났다.
    정작 Otsu 는 금괴를 2.2% 로 잘 잡고 있었다.

    카드를 빼서 아무것도 안 남으면 그건 카드가 아니었다는 뜻이다.
    """
    from app.pipeline import eratosthenes as er

    img = np.full((600, 800, 3), 40, np.uint8)
    cv2.rectangle(img, (330, 380), (470, 460), (60, 185, 230), -1)  # 금색 물체
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 물체를 통째로 덮는 가짜 카드 검출을 흉내 낸다
    bogus = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(bogus, (280, 330), (520, 510), 255, -1)
    original = er._card_exclusion
    er._card_exclusion = lambda _bgr, _s: (bogus, True)
    try:
        res = er.extract_outline(img)
    finally:
        er._card_exclusion = original

    assert res.meta["card_present"] is False, "가짜 카드를 진짜로 믿었다"
    ys, xs = np.where(res.mask > 0)
    assert 300 < xs.mean() < 500 and 350 < ys.mean() < 490, "물체를 못 찾았다"


def test_height_threshold_follows_measured_noise_both_ways():
    """
    높이 임계는 깊이 노이즈를 **양방향으로** 따라가야 한다.

    실측: 잘 찍힌 사진(RMSE 0.24mm)에서 종전 식 `max(2.0, 3·RMSE)` 는 상수 2.0 이
    바닥으로 남아, 높이 1.62mm 짜리 반지를 통째로 걸러 깊이 경로를 실패시켰다.
    그 결과 불안정한 외형 폴백으로 떨어져 같은 반지 마스크가 2.3%↔4.3% 로 흔들렸다.
    """
    from app.pipeline.height_segment import ABSOLUTE_MIN_HEIGHT_MM

    def threshold(rmse: float) -> float:
        return max(ABSOLUTE_MIN_HEIGHT_MM, 3.0 * rmse)

    # 좋은 사진: 임계가 내려가 얇은 물체도 잡힌다
    assert threshold(0.24) == pytest.approx(0.72)
    assert threshold(0.24) < 1.62, "높이 1.62mm 반지가 통과해야 한다"
    # 나쁜 사진: 임계가 알아서 올라가 노이즈를 물체로 착각하지 않는다
    assert threshold(1.089) == pytest.approx(3.267)
    assert threshold(1.089) > 1.285, "노이즈 수준 높이는 걸러야 한다"
    # 노이즈가 0 에 가까워도 절대 하한 아래로는 안 내려간다
    assert threshold(0.01) == pytest.approx(ABSOLUTE_MIN_HEIGHT_MM)
