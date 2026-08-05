from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings
from app.constants import ID1_HEIGHT_MM, ID1_WIDTH_MM
from app.pipeline.exceptions import PipelineError
from app.pipeline.precision_homography import evaluate_card_homography_precision

# ID-1 가로/세로 비 ≈ 1.586
_ID1_ASPECT = ID1_WIDTH_MM / ID1_HEIGHT_MM

# **겉보기** 종횡비 허용 범위. 원근 단축은 비율을 줄이기만 하므로 아래로 크게 연다.
# 실측(real5.jpg): 낮은 각도에서 찍힌 카드의 겉보기 비율이 1.095 였다 — 이전
# 하한 1.25 에 걸려 진짜 카드가 후보에서 탈락했고, 카드 인쇄물의 핑크색 절반이
# 대신 뽑혔다. 여기는 명백한 쓰레기만 거르는 1차 관문이고, 진짜 판별은
# `camera.implied_rectangle_aspect` 로 **역산한 비율**이 한다.
_ID1_APPARENT_MIN = 0.85
_ID1_APPARENT_MAX = 2.60
# 역산 비율이 1.586 에서 이보다 멀면 카드가 아니다(실측: 카드 1.676 / 골드바 2.156)
_ID1_IMPLIED_TOLERANCE = 0.40


@dataclass
class CardGeometry:
    sigma_mm_per_px: float
    quad_px: np.ndarray  # 4x2 float32 order TL,TR,BR,BL
    warped_preview: np.ndarray | None = None
    homography_3x3: np.ndarray | None = None  # getPerspectiveTransform (src image → card canonical)
    precision_pose_candidate: bool = False
    precision_solution_count: int = 0
    used_fallback_quad: bool = False
    long_edge_px: float = 0.0  # 원본 이미지에서의 카드 긴 변(ID-1 85.60mm 축) 평균 길이
    short_edge_px: float = 0.0  # 원본 이미지에서의 카드 짧은 변(53.98mm 축) 평균 길이


def card_edge_lengths_px(quad_px: np.ndarray) -> tuple[float, float]:
    """
    쿼드(TL,TR,BR,BL)의 마주보는 변 쌍을 평균해 (긴 변, 짧은 변) px 를 낸다.

    `order_quad_points` 는 sum/diff 휴리스틱이라 카드가 세로로 누우면 TL→TR 이
    짧은 변이 될 수 있다. ID-1 종횡비는 1.586 이라 길이로 판별하는 편이 안전하다.
    """
    q = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
    e_a = 0.5 * (float(np.linalg.norm(q[1] - q[0])) + float(np.linalg.norm(q[2] - q[3])))
    e_b = 0.5 * (float(np.linalg.norm(q[2] - q[1])) + float(np.linalg.norm(q[3] - q[0])))
    return max(e_a, e_b), min(e_a, e_b)


def sigma_mm_per_px_from_quad(quad_px: np.ndarray, view: str | None = None) -> float:
    """
    **원본 이미지 픽셀** 기준 mm/px.

    이전 구현은 워프 캔버스 폭(`dst_w=856`)으로 나눠 σ 가 항상 0.1 로 고정됐고,
    그 값이 원본 픽셀 좌표에 곱해져 단위가 어긋났다. 상세: `archimedes-v2-single-photo.mdc` §0.4.

    두 축에서 각각 σ 후보를 구한 뒤 **작은 쪽**을 택한다. 원근 단축(foreshortening)은
    투영 길이를 줄이기만 하므로(→ σ 를 크게만 만듦), 최솟값이 가장 덜 편향된 추정이다.
    """
    long_px, short_px = card_edge_lengths_px(quad_px)
    if long_px < 2.0 or short_px < 2.0:
        raise PipelineError(
            "ERR_CARD_NOT_FOUND",
            f"Degenerate card quad (long={long_px:.1f}px, short={short_px:.1f}px)",
            retry_step=view,
        )
    return min(ID1_WIDTH_MM / long_px, ID1_HEIGHT_MM / short_px)


# 카드가 프레임의 이 비율을 넘으면 물체가 함께 보일 수 없다 = 카드가 아니다.
# (강한 블러에서 **이미지 테두리 자체**가 사각형으로 잡히는 것을 막는다)
_MAX_CARD_AREA_RATIO = 0.45
# 프레임 가장자리에 이만큼 붙으면 잘린 것 — 카드 전체가 보여야 스케일이 성립한다
_BORDER_MARGIN_RATIO = 0.01


def _touches_border(quad: np.ndarray, shape: tuple[int, int, int] | tuple[int, int]) -> bool:
    h, w = shape[0], shape[1]
    m = max(2.0, _BORDER_MARGIN_RATIO * max(h, w))
    x, y = quad[:, 0], quad[:, 1]
    return bool(
        (x.min() <= m) or (y.min() <= m) or (x.max() >= w - 1 - m) or (y.max() >= h - 1 - m)
    )


def _quad_aspect(quad: np.ndarray) -> float:
    """쿼드의 (긴 변 / 짧은 변). ID-1 이면 원근 왜곡 전 1.586."""
    long_px, short_px = card_edge_lengths_px(quad)
    return long_px / max(short_px, 1e-6)


def _id1_aspect_error(
    quad: np.ndarray,
    shape: tuple[int, int, int] | tuple[int, int] | None,
    focal_px_hint: float | None = None,
) -> float:
    """
    ID-1(1.586)에서 얼마나 벗어났나. **역산 비율**을 쓰고, 못 풀면 겉보기로 낙하.

    겉보기 비율은 원근에 속는다 — 상세는 `camera.implied_rectangle_aspect` 참고.
    """
    if shape is not None:
        from app.pipeline.camera import implied_rectangle_aspect

        implied, _f = implied_rectangle_aspect(quad, int(shape[1]), int(shape[0]), focal_px_hint)
        if implied is not None:
            return abs(implied - _ID1_ASPECT)
    return abs(_quad_aspect(quad) - _ID1_ASPECT)


def _is_id1_like(
    quad: np.ndarray,
    shape: tuple[int, int, int] | tuple[int, int] | None = None,
    focal_px_hint: float | None = None,
) -> bool:
    """
    2단 관문: 겉보기 비율로 명백한 쓰레기를 거르고, 역산 비율로 진짜를 가른다.

    역산이 불가능한 쿼드(정면에 가까워 소실점이 무한대)는 겉보기만으로 통과시킨다 —
    정면일수록 겉보기 ≈ 실제라 겉보기 판정이 오히려 정확하다.
    """
    if not (_ID1_APPARENT_MIN <= _quad_aspect(quad) <= _ID1_APPARENT_MAX):
        return False
    if shape is None:
        return True
    from app.pipeline.camera import implied_rectangle_aspect

    implied, _f = implied_rectangle_aspect(quad, int(shape[1]), int(shape[0]), focal_px_hint)
    if implied is None:
        return True
    return abs(implied - _ID1_ASPECT) <= _ID1_IMPLIED_TOLERANCE


def _quad_score(
    quad: np.ndarray,
    area: float,
    img_area: float,
    shape: tuple[int, int, int] | tuple[int, int] | None = None,
    card_side: str = "any",
    focal_px_hint: float | None = None,
    fill: float = 1.0,
) -> float:
    """
    카드 후보 점수 = 크기 타당성 × ID-1 비율 근접도 × 사각형 충실도 × 위치 가점.

    **면적에 비례시키면 안 된다.** 그러면 카드가 옆 물체와 붙어 만들어진 병합
    덩어리가 언제나 이긴다 — 실측 real4 에서 카드는 중간 블러에서 채움 1.03 으로
    완벽히 잡혔는데도, 거친 블러의 병합 쿼드(36.4%)가 면적만으로 이겼다.
    카드의 정체는 **크기가 아니라 생김새**이므로 면적은 "충분히 크면 동등"으로
    포화시키고, 변별은 비율과 충실도에 맡긴다.

    비율은 **역산값**을 쓴다. 겉보기 비율로 점수를 매기면 낮은 각도에서 눌린
    카드(실측 1.095)가 골드바(2.079)보다 낮은 점수를 받아 뒤집힌다.
    """
    aspect_err = _id1_aspect_error(quad, shape, focal_px_hint) / _ID1_ASPECT
    # 프레임의 5% 를 넘으면 크기로는 더 우대하지 않는다
    size_term = min(1.0, (area / img_area) / 0.05)
    # 충실도는 세제곱 — 0.75(병합) 대 1.0(단독) 을 0.42 대 1.0 으로 벌린다
    fill_term = min(1.0, max(0.0, fill)) ** 3
    score = size_term * fill_term * (1.0 / (1.0 + 4.0 * aspect_err))

    if card_side in ("left", "right") and shape is not None:
        w = float(shape[1])
        # 프레임 중앙 기준 -1(왼쪽 끝) ~ +1(오른쪽 끝)
        rel = (float(quad[:, 0].mean()) - w / 2.0) / (w / 2.0)
        signed = rel if card_side == "right" else -rel
        # 규약대로면 최대 2배 가점, 반대편이면 감점
        score *= 1.0 + max(-0.7, min(1.0, signed))
    return score


def _largest_quad_from_contours(
    contours: list,
    shape: tuple[int, int, int] | tuple[int, int],
    min_area_ratio: float,
    eps_factors: tuple[float, ...] = (0.02, 0.035, 0.05),
    focal_px_hint: float | None = None,
) -> np.ndarray | None:
    """
    ID-1 비율에 맞는 사각형 중 **가장 큰 것**.

    ⚠️ 종횡비 검증이 없으면 프레임 안의 아무 사각형이나 카드로 잡는다.
    실측: 금괴(종횡비 2.149)를 신용카드(1.586)로 오인해 스케일이 통째로 어긋났다.
    귀금속 자체가 직사각형인 경우(골드바)가 흔하므로 반드시 걸러야 한다.
    """
    h = shape[0]
    w = shape[1]
    img_area = float(h * w)
    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps_f in eps_factors:
            approx = cv2.approxPolyDP(cnt, eps_f * peri, True)
            if len(approx) != 4:
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            if _quad_fill_ratio(area, quad) < _MIN_QUAD_FILL:
                break
            if not _is_id1_like(quad, shape, focal_px_hint):
                break
            if area > _MAX_CARD_AREA_RATIO * img_area or _touches_border(quad, shape):
                break
            # 면적만 보면 카드 인쇄가 만든 부분 사각형이 이길 수 있다.
            # ID-1 비율에 얼마나 가까운지를 함께 본다(실측: 카드의 초록 띠만
            # 잡아 종횡비 2.005 로 통과했다).
            score = _quad_score(quad, area, img_area, shape, focal_px_hint=focal_px_hint)
            if score > best_score:
                best_score = score
                best = quad
            break
    return best


def _min_area_rect_id1_like(
    cnt: np.ndarray,
    img_area: float,
    min_area_ratio: float,
    shape: tuple[int, int, int] | tuple[int, int] | None = None,
    focal_px_hint: float | None = None,
) -> np.ndarray | None:
    """
    윤곽이 4각 근사에 실패해도 카드 비율이면 채택.

    비율 판정은 **원근이 살아 있는 꼭짓점**으로 하고(역산 가능), 반환 쿼드도
    가능하면 그것을 준다. minAreaRect 를 그대로 돌려주면 이후 PnP·소실점이
    원근 없는 상자를 보게 돼 f 가 풀리지 않는다.
    """
    area = cv2.contourArea(cnt)
    if area < min_area_ratio * img_area or area > _MAX_CARD_AREA_RATIO * img_area:
        return None
    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), _ = rect
    if rw < 1.0 or rh < 1.0:
        return None
    quad = _perspective_quad(cnt)
    if quad is None:
        quad = cv2.boxPoints(rect).astype(np.float32)
    if _quad_fill_ratio(area, quad) < _MIN_QUAD_FILL:
        return None
    if not _is_id1_like(quad, shape, focal_px_hint):
        return None
    return quad


def _largest_id1_rect_from_contours(
    contours: list,
    shape: tuple[int, int, int] | tuple[int, int],
    min_area_ratio: float,
    focal_px_hint: float | None = None,
) -> np.ndarray | None:
    h = shape[0]
    w = shape[1]
    img_area = float(h * w)
    best = None
    best_area = 0.0
    for cnt in contours:
        quad = _min_area_rect_id1_like(cnt, img_area, min_area_ratio, shape, focal_px_hint)
        if quad is None or _touches_border(quad, shape):
            continue
        a = cv2.contourArea(cnt)
        if a > best_area:
            best_area = a
            best = quad
    return best


def _perspective_quad(cnt: np.ndarray) -> np.ndarray | None:
    """
    윤곽 → **원근이 살아 있는** 사각형 꼭짓점 4개.

    `cv2.minAreaRect` 는 정의상 회전 직사각형이라 원근 정보를 지운다. 그걸로는
    실제 비율을 역산할 수 없어(실측: 역산값이 겉보기와 똑같이 나옴) 카드와
    골드바를 못 가른다. 볼록껍질을 4각으로 근사해 진짜 꼭짓점을 얻는다.
    """
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    if peri <= 0:
        return None
    for eps_f in (0.005, 0.01, 0.015, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1):
        approx = cv2.approxPolyDP(hull, eps_f * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        if len(approx) < 4:
            break
    return None


def _quad_fill_ratio(filled_area: float, quad: np.ndarray) -> float:
    """
    덩어리가 **자기 쿼드를 얼마나 채우는가**. 카드 하나면 1 에 가깝다.

    이 값이 낮다 = 덩어리가 사각형이 아니다. 실측 실패 전부가 이 형태였다:
    카드가 옆의 밝은 물체와 **한 덩어리로 붙어** L 자가 되고, 그 볼록껍질을
    4각으로 근사하니 카드와 이웃을 함께 감싸는 큰 쿼드가 나왔다
    (real4: 카드+금괴+반사 병합 36.4%, real2: 카드+투명케이스 병합).
    비율만으로는 이런 쿼드가 통과한다 — 병합 결과가 우연히 1.5 근처가 되기 때문.

    ⚠️ `filled_area` 는 반드시 **실제 화소 수**여야 한다. `cv2.contourArea` 는
    외곽선이 감싸는 면적이라 가운데가 빈 병합 덩어리에서 빈 속까지 세어
    채움률을 부풀린다(실측: real4 병합 덩어리가 이 때문에 검사를 통과했다).
    """
    quad_area = abs(cv2.contourArea(np.asarray(quad, dtype=np.float32)))
    if quad_area < 1.0:
        return 0.0
    return float(filled_area / quad_area)


# 카드 덩어리는 자기 사각형을 이만큼은 채워야 한다(실측: 정상 0.93~0.98)
_MIN_QUAD_FILL = 0.82
# 다른 후보 안에 이만큼 들어가면 그 후보의 **부분 사각형**으로 본다
_CONTAINMENT_RATIO = 0.85


def _blob_candidate(
    cnt: np.ndarray,
    filled_area: float,
    shape: tuple[int, int, int] | tuple[int, int],
    img_area: float,
    min_area_ratio: float,
    focal_px_hint: float | None,
) -> tuple[np.ndarray, float] | None:
    """윤곽 하나 → 검증을 통과한 (카드 후보 쿼드, 사각형 충실도)."""
    if filled_area < min_area_ratio * img_area or filled_area > _MAX_CARD_AREA_RATIO * img_area:
        return None
    quad = _perspective_quad(cnt)
    if quad is None:
        # 4각 근사 실패 시에만 회전사각형으로 낙하 — 역산은 못 하지만
        # 후보가 아예 없는 것보단 낫다
        quad = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
    fill = _quad_fill_ratio(filled_area, quad)
    if fill < _MIN_QUAD_FILL:
        return None
    if not _is_id1_like(quad, shape, focal_px_hint) or _touches_border(quad, shape):
        return None
    return quad, fill


def _local_fill(gray: np.ndarray, quad: np.ndarray) -> float:
    """
    쿼드 **안쪽이 실제로 한 덩어리인가**를 국소 이진화로 잰다.

    `_quad_fill_ratio` 는 밝기 덩어리 경로에서만 쓸 수 있다. 엣지(Canny) 경로의
    쿼드는 윤곽선 자체가 곧 쿼드라 충실도를 잴 수가 없어 1.0 을 주고 있었는데,
    그 공짜 점수 때문에 **카드와 옆 물체를 함께 감싼 쿼드**가 그대로 통과했다
    (실측 real2: 카드+투명케이스를 감싼 연 모양이 채택돼 σ 가 어긋났다).

    전역 Otsu 는 못 쓴다 — 모니터·키보드가 함께 밝으면 밝은 쪽이 프레임의
    절반을 넘어 후보가 통째로 사라진다(실측 real2 에서 밝기 후보 0개).
    쿼드 주변만 잘라 그 안에서 이진화하면 배경이 무엇이든 성립한다.
    """
    h, w = gray.shape[:2]
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    x0 = max(0, int(np.floor(q[:, 0].min())))
    y0 = max(0, int(np.floor(q[:, 1].min())))
    x1 = min(w, int(np.ceil(q[:, 0].max())) + 1)
    y1 = min(h, int(np.ceil(q[:, 1].max())) + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    poly = (q - [x0, y0]).astype(np.int32)
    inside = np.zeros(patch.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, poly, 255)
    n_inside = int(cv2.countNonZero(inside))
    if n_inside < 64:
        return 0.0

    _t, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 쿼드 안쪽에서 다수를 차지하는 쪽이 '물체'다 — 밝은 카드든 어두운 카드든 성립
    bright = int(cv2.countNonZero(cv2.bitwise_and(binary, inside)))
    coherence = float(max(bright, n_inside - bright)) / float(n_inside)

    # 안쪽이 균일하기만 하면 **빈 책상 한 조각도 만점**을 받는다. 실측 real2 에서
    # 적응형 경로가 모니터까지 걸친 큰 쿼드를 냈는데 안쪽이 고르게 어두워 통과했다.
    # 진짜 물체 경계라면 바로 바깥 띠와 밝기가 달라야 한다.
    short_px = min(card_edge_lengths_px(q))
    band = max(3, round(short_px * 0.06))
    outer = cv2.dilate(inside, np.ones((band * 2 + 1, band * 2 + 1), np.uint8))
    ring = cv2.subtract(outer, inside)
    if int(cv2.countNonZero(ring)) < 32:
        return coherence
    med_in = float(np.median(patch[inside > 0]))
    med_out = float(np.median(patch[ring > 0]))
    # 20 계조(8bit 의 8%)면 충분한 경계로 본다
    contrast_term = min(1.0, abs(med_in - med_out) / 20.0)
    return coherence * contrast_term


def _shape_quality(
    quad: np.ndarray,
    fill: float,
    shape: tuple[int, int, int] | tuple[int, int] | None,
    focal_px_hint: float | None,
) -> float:
    """크기를 뺀 '카드다움' — 사각형 충실도 × ID-1 비율 근접도."""
    aspect_err = _id1_aspect_error(quad, shape, focal_px_hint) / _ID1_ASPECT
    return (min(1.0, max(0.0, fill)) ** 3) * (1.0 / (1.0 + 4.0 * aspect_err))


def _suppress_contained(
    cands: list[tuple[np.ndarray, float]],
    shape: tuple[int, int, int] | tuple[int, int] | None = None,
    focal_px_hint: float | None = None,
) -> list[tuple[np.ndarray, float]]:
    """
    다른 후보에 거의 통째로 들어가는 쿼드를 버린다. **카드는 카드다운 것 중 가장 바깥.**

    카드 **인쇄물이 만든 부분 사각형**(핑크 절반, 초록 띠)은 정의상 카드 안에
    있고, 인쇄 경계라 비율이 우연히 1.586 에 더 가까울 수도 있다. 실측 real4 에서
    비율만 보게 했더니 초록 절반(역산 1.601)이 카드 전체(1.826)를 이겼다.
    포함 관계로 가르면 이 부류가 통째로 사라진다.

    단, 바깥 쿼드가 **병합 덩어리**일 수 있으므로(카드+금괴) 무조건 바깥을
    택하지는 않는다. 바깥의 카드다움이 안쪽의 60% 밑으로 떨어지면 억제하지 않는다.

    ⚠️ 이 억제는 **모든 블러 단계의 후보를 모은 뒤** 한 번에 해야 한다. 단계별로
    따로 하면 다른 단계에서 나온 전체 카드와 비교가 안 돼 부분 사각형이 살아남는다.
    """
    keep: list[tuple[np.ndarray, float]] = []
    areas = [abs(cv2.contourArea(q.astype(np.float32))) for q, _f in cands]
    quals = [_shape_quality(q, f, shape, focal_px_hint) for q, f in cands]
    for i, (qi, fi) in enumerate(cands):
        contained = False
        for j, (qj, _fj) in enumerate(cands):
            if i == j or areas[j] <= areas[i]:
                continue
            if quals[j] < 0.6 * quals[i]:
                continue  # 바깥이 훨씬 덜 카드답다 = 병합 덩어리로 본다
            # 꼭짓점 개수로 세면 한 점만 살짝 밖으로 나가도 포함이 깨진다
            # (실측 real4: 초록 절반의 좌우 꼭짓점이 카드 외곽선에 닿아 억제 실패).
            # 면적 겹침으로 재면 경계의 1~2px 오차에 흔들리지 않는다.
            inter, _pts = cv2.intersectConvexConvex(
                qi.astype(np.float32), qj.astype(np.float32)
            )
            if areas[i] > 1.0 and inter / areas[i] >= _CONTAINMENT_RATIO:
                contained = True
                break
        if not contained:
            keep.append((qi, fi))
    return keep or cands


def background_contrast_binary(bgr: np.ndarray) -> np.ndarray:
    """
    **배경(바닥) 색에서 먼 화소**를 전경으로 본다. 카드가 한 덩어리가 된다.

    왜 밝기로는 안 되는가: 신용카드는 인쇄가 밝기까지 갈라 놓는다. 실측
    (도련님 08:25 사진) 에서 카드 위쪽 핑크·빨강 영역이 Otsu 임계 아래로 떨어져
    **초록 아래쪽 절반만** 덩어리가 됐고, 전체 카드는 후보에 아예 오르지 못했다.
    엣지·적응형도 같은 경계를 외곽선으로 잡아 전부 초록 절반을 냈다.
    그 결과 σ 가 1.3배 어긋나 금괴가 52.4×24.1mm(실제 약 40×20)로 나왔다.

    배경색 기준으로 재면 핑크든 초록이든 흰색이든 "바닥이 아닌 것"으로 묶인다.
    배경색은 **프레임 테두리 띠의 중앙값**으로 잡는다 — 촬영 규약상 물체와 카드는
    가운데 있고 테두리는 바닥이다.

    실측 개선: 도련님 사진에서 전체 카드(프레임의 15.8%, 역산 비율 1.623)를
    처음으로 후보에 올렸다.
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    m = max(4, round(min(h, w) * 0.06))
    ring = np.concatenate(
        [
            lab[:m].reshape(-1, 3),
            lab[-m:].reshape(-1, 3),
            lab[:, :m].reshape(-1, 3),
            lab[:, -m:].reshape(-1, 3),
        ]
    )
    bg = np.median(ring, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = (255.0 * dist / max(float(dist.max()), 1e-6)).astype(np.uint8)
    _t, binary = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, round(max(h, w) * 0.004) | 1)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))


def _split_and_extract(
    binary: np.ndarray,
    gray: np.ndarray,
    shape: tuple[int, int, int] | tuple[int, int],
    min_area_ratio: float,
    focal_px_hint: float | None,
) -> list[tuple[np.ndarray, float]]:
    """이진 마스크 → 카드 후보들. 침식으로 병합 덩어리를 떼고 검증까지 한다."""
    h, w = shape[0], shape[1]
    img_area = float(h * w)
    k = max(3, round(max(h, w) * 0.004) | 1)

    # 카드가 옆의 밝은 물체(금괴·투명케이스)와 한 덩어리로 붙는 일이 잦다.
    # **침식으로 가는 연결을 끊고**, 끊어진 조각을 다시 팽창시켜 원래 모양을
    # 되찾은 뒤 후보로 삼는다. 침식 없이 한 번만 보면 병합 덩어리밖에 못 본다.
    ker = np.ones((k, k), np.uint8)
    out: list[tuple[np.ndarray, float]] = []
    for erode_it in (0, 1, 2, 3):
        eroded = binary if erode_it == 0 else cv2.erode(binary, ker, iterations=erode_it)
        n, labels, _stats, _c = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        if n <= 1:
            break
        for i in range(1, n):
            seed = ((labels == i).astype(np.uint8)) * 255
            if erode_it:
                # 침식한 만큼(+1) 되돌리고 원본 덩어리로 잘라 경계를 복원한다
                seed = cv2.bitwise_and(cv2.dilate(seed, ker, iterations=erode_it + 1), binary)
            cnts, _h = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cand = _blob_candidate(
                max(cnts, key=cv2.contourArea),
                float(cv2.countNonZero(seed)),
                shape,
                img_area,
                min_area_ratio,
                focal_px_hint,
            )
            if cand is not None:
                # 전역 이진화가 놓친 병합도 국소 이진화로 한 번 더 본다
                cand_quad, blob_fill = cand
                out.append((cand_quad, min(blob_fill, _local_fill(gray, cand_quad))))
    # 억제는 호출부에서 **모든 블러 단계를 모은 뒤** 한 번에 한다
    return out


def _bright_blob_quads(
    gray: np.ndarray,
    shape: tuple[int, int, int] | tuple[int, int],
    min_area_ratio: float,
    focal_px_hint: float | None = None,
) -> list[tuple[np.ndarray, float]]:
    """
    **밝은 덩어리 통째로** 카드 후보를 만든다.

    엣지 기반은 카드 **인쇄물의 색 경계**(핑크/초록)를 외곽선과 구분하지 못해
    카드를 조각낸다(실측: 초록 띠만 잡힘). 밝기로 묶으면 인쇄 경계 일부는 넘어선다.
    다만 인쇄가 **밝기까지** 갈라 놓으면 이것도 진다 — 그때는
    `background_contrast_binary` 경로가 받는다.
    """
    _t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(binary.mean()) > 127:  # 배경이 밝으면 반전
        binary = 255 - binary
    k = max(3, round(max(shape[0], shape[1]) * 0.004) | 1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return _split_and_extract(binary, gray, shape, min_area_ratio, focal_px_hint)


def _bg_contrast_quads(
    bgr: np.ndarray,
    gray: np.ndarray,
    shape: tuple[int, int, int] | tuple[int, int],
    min_area_ratio: float,
    focal_px_hint: float | None = None,
) -> list[tuple[np.ndarray, float]]:
    """배경색 대비로 카드 후보를 만든다 — 인쇄 색·밝기와 무관하게 한 덩어리."""
    return _split_and_extract(
        background_contrast_binary(bgr), gray, shape, min_area_ratio, focal_px_hint
    )


def _contours_from_canny(gray: np.ndarray, t1: int, t2: int) -> list:
    edges = cv2.Canny(gray, t1, t2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)[:30]


def _contours_from_adaptive(gray: np.ndarray) -> list:
    """
    국소 대비로 카드 외곽선을 찾는다. **전역 밝기에 흔들리지 않는다.**

    실측 real2: 모니터·키보드가 함께 밝아 전역 Otsu 의 밝은 쪽이 프레임 절반을
    넘어 밝기 덩어리 후보가 0 개였고, 카드 자체도 어두워 Canny 는 카드 외곽선
    대신 **인쇄물의 핑크/초록 경계**만 잡았다. 그래서 카드 아래 절반이 채택됐다.
    적응형 이진화는 두 조건 어디에도 걸리지 않는다.
    """
    h, w = gray.shape[:2]
    # 블록은 카드 한 변보다 작아야 카드 안팎의 대비를 본다
    block = max(11, (round(min(h, w) * 0.02) | 1))
    out: list = []
    for c_off in (3, 8):
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c_off
        )
        k = max(3, round(max(h, w) * 0.003) | 1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        cnts, _hier = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out.extend(sorted(cnts, key=cv2.contourArea, reverse=True)[:20])
    return out


def detect_card_quad(
    bgr: np.ndarray,
    view: str,
    card_side: str = "any",
    focal_px_hint: float | None = None,
) -> np.ndarray:
    """
    프레임에서 신용카드(ID-1) 쿼드를 찾는다.

    **블러 크기를 해상도에 비례**시킨다. 4032px 사진에 5px 블러는 사실상 없는
    것과 같아, 카드 **인쇄물의 색 경계**(핑크/초록 등)가 외곽선만큼 뚜렷하게
    잡히고 카드가 조각난다.
    실측: 카드의 초록 띠만 쿼드로 잡혀 스케일이 어긋났다.
    """
    base = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    long_side = max(bgr.shape[:2])
    img_area = float(bgr.shape[0] * bgr.shape[1])

    # 블러 하나로는 안 된다. 작으면 카드 인쇄가 조각으로 잡히고, 크면 카드
    # 외곽선까지 뭉개져 아무것도 못 찾는다(실측 양쪽 다 발생).
    # 여러 스케일을 모두 시도하고 **가장 카드다운 후보**를 고른다.
    blur_ks = sorted({max(5, round(long_side * r) | 1) for r in (0.0012, 0.0025, 0.005)})

    quad = None
    best_score = -1.0
    pooled: list[tuple[np.ndarray, float]] = []
    for k in blur_ks:
        gray = cv2.GaussianBlur(base, (k, k), 0)
        contour_sets = [_contours_from_canny(gray, t1, t2) for t1, t2 in ((40, 120), (30, 90), (25, 80))]
        # 적응형 이진화는 **정규 후보원**이다(폴백이 아니다). 어두운 카드·밝은 배경
        # 조합에서 Canny·전역Otsu 가 둘 다 실패해도 여기서는 외곽선이 나온다.
        contour_sets.append(_contours_from_adaptive(gray))
        for contours in contour_sets:
            # 엣지 경로는 윤곽선이 곧 쿼드라 덩어리 채움률을 못 잰다 → 국소 이진화로 잰다
            pooled.extend(
                (c, _local_fill(gray, c))
                for c in (
                    _largest_quad_from_contours(
                        contours, bgr.shape, min_area_ratio=0.03, focal_px_hint=focal_px_hint
                    ),
                    _largest_id1_rect_from_contours(
                        contours, bgr.shape, min_area_ratio=0.03, focal_px_hint=focal_px_hint
                    ),
                )
                if c is not None
            )
        pooled.extend(
            _bright_blob_quads(gray, bgr.shape, min_area_ratio=0.03, focal_px_hint=focal_px_hint)
        )

    # 배경색 대비는 블러 단계와 무관하므로 한 번만 돈다
    pooled.extend(
        _bg_contrast_quads(bgr, base, bgr.shape, min_area_ratio=0.03, focal_px_hint=focal_px_hint)
    )

    for cand, fill in _suppress_contained(pooled, bgr.shape, focal_px_hint):
        sc = _quad_score(
            cand,
            cv2.contourArea(cand.astype(np.float32)),
            img_area,
            bgr.shape,
            card_side,
            focal_px_hint,
            fill,
        )
        if sc > best_score:
            best_score, quad = sc, cand

    if quad is None:
        gray = cv2.GaussianBlur(base, (blur_ks[0], blur_ks[0]), 0)
        # Fallback: adaptive threshold for high-contrast card on desk
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5
        )
        contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        quad = _largest_quad_from_contours(
            contours, bgr.shape, min_area_ratio=0.02, focal_px_hint=focal_px_hint
        )
        if quad is None:
            quad = _largest_id1_rect_from_contours(
                contours, bgr.shape, min_area_ratio=0.02, focal_px_hint=focal_px_hint
            )

    if quad is None:
        raise PipelineError(
            "ERR_CARD_NOT_FOUND",
            f"'{view}' 컷에서 신용카드를 찾지 못했습니다. 카드 전체가 잘리지 않게, "
            "배경과 구분되도록(어두운 바닥 위 등) 다시 찍어 주세요.",
            retry_step=view,
            error_severity="soft",
            suggested_action="retry_one_view",
        )
    return order_quad_points(quad)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    TL, TR, BR, BL 순으로 정렬한다.

    **중심각 기준 순환 정렬**을 쓴다. 이전 구현은 좌표합(x+y)·차(y−x)의
    argmin/argmax 로 네 꼭짓점을 골랐는데, 카드가 45도 부근으로 기울면
    **같은 점이 두 번 뽑혀** 쿼드가 붕괴한다.

    실측(도련님 사진): 45도 기운 카드에서 4점 중 1점이 중복되어
    한 변의 길이가 0 이 되고, 종횡비가 2.087(정답 1.586)로 나왔다.
    그 결과 σ·소실점 초점거리·PnP 가 전부 어긋났다.
    각도로 정렬하면 순환 순서가 보장돼 중복이 원천적으로 불가능하다.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    center = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0])
    p = p[np.argsort(ang)]

    # 이미지 좌표(y 아래)에서 TL→TR→BR→BL 은 shoelace 가 양수인 방향이다
    x, y = p[:, 0], p[:, 1]
    if float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
        p = p[::-1]

    # 좌상단에 가장 가까운 점에서 시작하도록 순환 이동
    start = int(np.argmin(p[:, 0] + p[:, 1]))
    p = np.roll(p, -start, axis=0)
    return p.astype(np.float32)


def _fallback_quad_from_frame(shape: tuple[int, int, int] | tuple[int, int]) -> np.ndarray:
    """카드 미검출 시 프레임 중심에 ID-1 비율 사각형을 가정해 파이프라인 중단을 피한다."""
    h = float(shape[0])
    w = float(shape[1])
    target_ratio = ID1_WIDTH_MM / ID1_HEIGHT_MM
    max_w = w * 0.78
    max_h = h * 0.78
    fw = max_w
    fh = fw / target_ratio
    if fh > max_h:
        fh = max_h
        fw = fh * target_ratio
    cx = w * 0.5
    cy = h * 0.5
    tl = [cx - fw * 0.5, cy - fh * 0.5]
    tr = [cx + fw * 0.5, cy - fh * 0.5]
    br = [cx + fw * 0.5, cy + fh * 0.5]
    bl = [cx - fw * 0.5, cy + fh * 0.5]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_card_and_sigma(bgr: np.ndarray, quad: np.ndarray, view: str | None = None) -> CardGeometry:
    """
    카드 정면 워프(**미리보기·호모그래피 전용**) + 원본 픽셀 기준 σ 산출.

    워프는 시각 검수와 `decomposeHomographyMat` 입력으로만 쓰고,
    스케일은 반드시 **원본 쿼드**에서 뽑는다(`sigma_mm_per_px_from_quad`).
    """
    dst_w = 856
    dst_h = round(dst_w * (ID1_HEIGHT_MM / ID1_WIDTH_MM))
    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr, H, (dst_w, dst_h))
    sigma = sigma_mm_per_px_from_quad(quad, view)
    long_px, short_px = card_edge_lengths_px(quad)
    prec_ok, prec_n = evaluate_card_homography_precision(H, bgr.shape)
    return CardGeometry(
        sigma_mm_per_px=sigma,
        quad_px=quad,
        warped_preview=warped,
        homography_3x3=H,
        precision_pose_candidate=prec_ok,
        precision_solution_count=prec_n,
        long_edge_px=long_px,
        short_edge_px=short_px,
    )


def _card_side_from(settings: Settings | None) -> str:
    """물체가 왼쪽이면 카드는 오른쪽 — 규약의 반대편."""
    side = (getattr(settings, "object_side", "any") or "any").strip().lower()
    return {"left": "right", "right": "left"}.get(side, "any")


def compute_card_geometry(
    bgr: np.ndarray,
    view: str,
    settings: Settings | None = None,
    exif: dict | None = None,
) -> CardGeometry:
    """
    카드(메트릭 앵커) 검출 + σ 산출.

    v2에서 카드는 **선택적 앵커**다. 검출 실패 시 날조된 사각형으로 진행하는 대신
    깊이추정 단독 경로(+ tier 캡)로 넘어가는 것이 원칙이다(`archimedes-v2-single-photo.mdc` §3.2).
    `allow_card_fallback` 은 데모 전용 완화 스위치이며 기본 OFF.

    `exif` 를 주면 검출 단계에서 **비율 역산**에 EXIF 초점거리를 쓴다. 카드가
    정면에 가까워 소실점이 안 풀릴 때 판정 정확도가 올라간다.
    """
    from app.pipeline.camera import intrinsics_from_exif

    h, w = bgr.shape[:2]
    f_hint = intrinsics_from_exif(exif, w, h).fx
    try:
        quad = detect_card_quad(bgr, view, _card_side_from(settings), f_hint)
        geom = warp_card_and_sigma(bgr, quad, view)
        geom.used_fallback_quad = False
        return geom
    except PipelineError:
        if settings is None or not settings.allow_card_fallback:
            raise
        # 완화 모드: 카드 미검출이어도 임시 사각형으로 진행 — 스케일이 날조되므로 데모 외 사용 금지
        quad = _fallback_quad_from_frame(bgr.shape)
        geom = warp_card_and_sigma(bgr, quad, view)
        geom.used_fallback_quad = True
        return geom


def try_compute_card_geometry(
    bgr: np.ndarray,
    view: str,
    settings: Settings | None = None,
    exif: dict | None = None,
) -> CardGeometry | None:
    """앵커가 선택적인 v2 단일사진 경로용 — 검출 실패를 예외 대신 None 으로."""
    try:
        return compute_card_geometry(bgr, view, settings, exif)
    except PipelineError:
        return None
