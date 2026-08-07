"""
카드 판별 회귀 테스트.

실측 실패를 그대로 재현한다. 도련님 사진에서 반복해서 터진 두 가지다.

1. **원근에 눌린 카드**: 낮은 각도로 찍으면 겉보기 종횡비가 1.1 근처까지
   눌린다. 겉보기만 보면 카드가 후보에서 탈락하고 **카드 인쇄물의 핑크색
   절반**이 대신 뽑힌다. 역산 비율(`implied_rectangle_aspect`)은 안 속는다.
2. **부분 사각형**: 카드 안의 인쇄 경계는 비율이 우연히 1.586 에 더 가까울
   수 있다. 포함 관계로 걸러야 한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.camera import implied_rectangle_aspect
from app.pipeline.card import _suppress_contained, order_quad_points

ID1 = 85.60 / 53.98


def _project(corners_mm: np.ndarray, rvec, tvec, f: float, w: int, h: int) -> np.ndarray:
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
    pts, _ = cv2.projectPoints(corners_mm, rvec, tvec, K, np.zeros((4, 1)))
    return order_quad_points(pts.reshape(-1, 2).astype(np.float32))


def _rect_corners(width_mm: float, height_mm: float) -> np.ndarray:
    hx, hy = width_mm / 2.0, height_mm / 2.0
    return np.array(
        [[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]], np.float64
    )


def test_steeply_tilted_card_recovers_id1_aspect():
    """겉보기는 눌려도 역산 비율은 1.586 근처로 돌아온다."""
    w, h, f = 3024, 4032, 4600.0
    # 두 축으로 기울여 촬영 — 원근 단축이 크게 걸리고 소실점이 둘 다 유한하다.
    # (한 축으로만 기울이면 나머지 방향의 변이 평행으로 남아 f 가 안 풀린다)
    rvec = np.array([[np.deg2rad(58.0)], [np.deg2rad(28.0)], [0.0]])
    tvec = np.array([[0.0], [0.0], [300.0]])
    quad = _project(_rect_corners(85.60, 53.98), rvec, tvec, f, w, h)

    e_a = np.linalg.norm(quad[1] - quad[0])
    e_b = np.linalg.norm(quad[2] - quad[1])
    apparent = max(e_a, e_b) / min(e_a, e_b)
    implied, f_solved = implied_rectangle_aspect(quad, w, h, f)

    # 겉보기는 정답에서 크게 벗어나 있어야 한다(= 이 테스트가 의미 있는 상황)
    assert abs(apparent - ID1) > 0.25, f"apparent={apparent}"
    # 역산은 정답을 되찾는다
    assert implied is not None
    assert abs(implied - ID1) < 0.05, f"implied={implied}"
    # 기울었으니 소실점으로 초점거리도 풀려야 한다
    assert f_solved is not None
    assert abs(f_solved - f) / f < 0.05


def test_gold_bar_is_not_mistaken_for_card():
    """같은 자세로 놓인 골드바(2.5:1)는 역산 비율로 카드와 갈린다."""
    w, h, f = 3024, 4032, 4600.0
    rvec = np.array([[np.deg2rad(55.0)], [0.0], [np.deg2rad(15.0)]])
    tvec = np.array([[0.0], [0.0], [300.0]])

    card = _project(_rect_corners(85.60, 53.98), rvec, tvec, f, w, h)
    bar = _project(_rect_corners(40.0, 16.0), rvec, tvec, f, w, h)

    card_implied, _ = implied_rectangle_aspect(card, w, h, f)
    bar_implied, _ = implied_rectangle_aspect(bar, w, h, f)

    assert card_implied is not None and bar_implied is not None
    assert abs(card_implied - ID1) < 0.05
    assert abs(bar_implied - ID1) > 0.5, f"bar={bar_implied}"


def test_min_area_rect_destroys_perspective_evidence():
    """
    `minAreaRect` 박스로는 역산이 성립하지 않는다 — 회귀 방지.

    실측에서 이걸 넣는 바람에 역산값이 겉보기와 똑같이 나와 판별이 죽었다.
    """
    w, h, f = 3024, 4032, 4600.0
    rvec = np.array([[np.deg2rad(60.0)], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [300.0]])
    quad = _project(_rect_corners(85.60, 53.98), rvec, tvec, f, w, h)

    box = cv2.boxPoints(cv2.minAreaRect(quad.astype(np.float32))).astype(np.float32)
    box_implied, box_f = implied_rectangle_aspect(box, w, h, f)

    # 축정렬 상자는 마주보는 변이 평행이라 f 가 안 풀린다
    assert box_f is None
    quad_implied, _ = implied_rectangle_aspect(quad, w, h, f)
    assert quad_implied is not None
    assert abs(quad_implied - ID1) < abs((box_implied or 99.0) - ID1)


def test_printed_sub_rectangle_is_suppressed():
    """카드 안의 인쇄 사각형이 비율은 더 좋아도 포함 관계로 버려진다."""
    w, h, f = 3024, 4032, 4600.0
    rvec = np.array([[np.deg2rad(55.0)], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [300.0]])
    card = _project(_rect_corners(85.60, 53.98), rvec, tvec, f, w, h)
    # 카드 안쪽 인쇄 — 실제 비율은 오히려 정확히 1.586
    inner = _project(_rect_corners(60.0, 37.83), rvec, tvec, f, w, h)

    kept = _suppress_contained([(card, 1.0), (inner, 1.0)], (h, w), f)
    kept_areas = [abs(cv2.contourArea(q.astype(np.float32))) for q, _ in kept]
    card_area = abs(cv2.contourArea(card.astype(np.float32)))

    assert len(kept) == 1
    assert abs(kept_areas[0] - card_area) < card_area * 0.01


def test_merged_blob_does_not_suppress_the_card():
    """
    카드 + 옆 물체가 한 덩어리로 잡힌 쿼드는 카드를 억제하지 못한다.

    바깥이라고 무조건 이기면 병합 덩어리가 카드를 삼킨다(실측 real4: 36.4%).
    카드다움이 크게 떨어지면 억제를 포기해야 한다.
    """
    w, h, f = 3024, 4032, 4600.0
    rvec = np.array([[np.deg2rad(55.0)], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [300.0]])
    card = _project(_rect_corners(85.60, 53.98), rvec, tvec, f, w, h)
    merged = _project(_rect_corners(180.0, 150.0), rvec, tvec, f, w, h)

    # 병합 덩어리는 안이 성기다(채움 0.6) — 카드는 꽉 찬다
    kept = _suppress_contained([(card, 0.97), (merged, 0.60)], (h, w), f)
    areas = [abs(cv2.contourArea(q.astype(np.float32))) for q, _ in kept]
    card_area = abs(cv2.contourArea(card.astype(np.float32)))

    assert any(abs(a - card_area) < card_area * 0.01 for a in areas)


def test_background_contrast_groups_a_two_tone_card():
    """
    인쇄가 **밝기까지** 갈라 놓은 카드도 배경색 대비로는 한 덩어리가 된다.

    실측(도련님 08:25 사진): 카드 위쪽 핑크·빨강이 Otsu 임계 아래로 떨어져
    초록 아래 절반만 덩어리가 됐고, 전체 카드는 후보에 오르지도 못했다.
    """
    from app.pipeline.card import background_contrast_binary

    h, w = 900, 700
    img = np.full((h, w, 3), 45, np.uint8)  # 어두운 책상
    # 위 절반은 어두운 빨강(밝기가 배경과 가깝다), 아래 절반은 밝은 초록
    img[300:520, 200:560] = (40, 40, 165)
    img[520:740, 200:560] = (80, 220, 90)

    binary = background_contrast_binary(img)
    n, labels, stats, _c = cv2.connectedComponentsWithStats(binary, connectivity=8)
    assert n > 1
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = labels == biggest

    # 두 색 영역이 **같은** 성분에 들어가야 한다
    assert comp[400, 380], "위쪽(어두운 빨강)이 전경에 없다"
    assert comp[640, 380], "아래쪽(밝은 초록)이 전경에 없다"
    ys, _xs = np.where(comp)
    assert ys.min() < 340 and ys.max() > 700, "카드 전체 높이를 못 덮었다"
