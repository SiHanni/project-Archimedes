from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings
from app.constants import ID1_HEIGHT_MM, ID1_WIDTH_MM
from app.pipeline.exceptions import PipelineError
from app.pipeline.precision_homography import evaluate_card_homography_precision

# ID-1 가로/세로 비 ≈ 1.586 — 실사진에서 왜곡·원근으로 범위를 넓게 둠
_ID1_ASPECT = ID1_WIDTH_MM / ID1_HEIGHT_MM
_ID1_ASPECT_MIN = 1.25
_ID1_ASPECT_MAX = 2.05


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


def _is_id1_like(quad: np.ndarray) -> bool:
    return _ID1_ASPECT_MIN <= _quad_aspect(quad) <= _ID1_ASPECT_MAX


def _quad_score(
    quad: np.ndarray,
    area: float,
    img_area: float,
    shape: tuple[int, int, int] | tuple[int, int] | None = None,
    card_side: str = "any",
) -> float:
    """
    카드 후보 점수 = 면적 비중 × ID-1 비율 근접도 × 촬영 규약 위치 가점.

    면적만 보면 카드 **인쇄물이 만든 부분 사각형**이 이길 수 있고
    (실측: 초록 띠만 잡혀 종횡비 2.005), 비율만 보면 **귀금속 자체**가 이길 수 있다
    (실측: 골드바를 카드로 오인).

    결정적인 단서는 **촬영 규약**이다. "귀금속 왼쪽 · 카드 오른쪽"을 지키면
    두 밝은 사각형이 위치로 갈린다. 생김새만으로는 신용카드와 골드바를 구분할
    수 없으므로, 규약이 있는 쪽을 쓰는 편이 훨씬 견고하다.
    """
    aspect_err = abs(_quad_aspect(quad) - _ID1_ASPECT) / _ID1_ASPECT
    score = (area / img_area) * (1.0 / (1.0 + 4.0 * aspect_err))

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
            if not _is_id1_like(quad):
                break
            if area > _MAX_CARD_AREA_RATIO * img_area or _touches_border(quad, shape):
                break
            # 면적만 보면 카드 인쇄가 만든 부분 사각형이 이길 수 있다.
            # ID-1 비율에 얼마나 가까운지를 함께 본다(실측: 카드의 초록 띠만
            # 잡아 종횡비 2.005 로 통과했다).
            score = _quad_score(quad, area, img_area)
            if score > best_score:
                best_score = score
                best = quad
            break
    return best


def _min_area_rect_id1_like(
    cnt: np.ndarray,
    img_area: float,
    min_area_ratio: float,
) -> np.ndarray | None:
    """윤곽이 4각 근사에 실패해도, 최소면적 회전사각형이 카드 비율이면 채택."""
    area = cv2.contourArea(cnt)
    if area < min_area_ratio * img_area or area > _MAX_CARD_AREA_RATIO * img_area:
        return None
    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), _ = rect
    if rw < 1.0 or rh < 1.0:
        return None
    ar = max(rw, rh) / min(rw, rh)
    if not (_ID1_ASPECT_MIN <= ar <= _ID1_ASPECT_MAX):
        return None
    return cv2.boxPoints(rect).astype(np.float32)


def _largest_id1_rect_from_contours(
    contours: list,
    shape: tuple[int, int, int] | tuple[int, int],
    min_area_ratio: float,
) -> np.ndarray | None:
    h = shape[0]
    w = shape[1]
    img_area = float(h * w)
    best = None
    best_area = 0.0
    for cnt in contours:
        quad = _min_area_rect_id1_like(cnt, img_area, min_area_ratio)
        if quad is None or _touches_border(quad, shape):
            continue
        a = cv2.contourArea(cnt)
        if a > best_area:
            best_area = a
            best = quad
    return best


def _bright_blob_quads(
    gray: np.ndarray,
    shape: tuple[int, int, int] | tuple[int, int],
    min_area_ratio: float,
) -> list[np.ndarray]:
    """
    **밝은 덩어리 통째로** 카드 후보를 만든다.

    엣지 기반은 카드 **인쇄물의 색 경계**(핑크/초록)를 외곽선과 구분하지 못해
    카드를 조각낸다(실측: 초록 띠만 잡힘). 반면 우리 프로토콜은 "어두운 바닥"을
    요구하므로, 밝기로 이진화하면 카드는 색과 무관하게 **한 덩어리**가 된다.
    """
    h, w = shape[0], shape[1]
    img_area = float(h * w)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(binary.mean()) > 127:  # 배경이 밝으면 반전
        binary = 255 - binary
    k = max(3, round(max(h, w) * 0.004) | 1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[np.ndarray] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_ratio * img_area or area > _MAX_CARD_AREA_RATIO * img_area:
            continue
        comp = ((labels == i).astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(max(cnts, key=cv2.contourArea))).astype(np.float32)
        if _is_id1_like(box) and not _touches_border(box, shape):
            out.append(box)
    return out


def _contours_from_canny(gray: np.ndarray, t1: int, t2: int) -> list:
    edges = cv2.Canny(gray, t1, t2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)[:30]


def detect_card_quad(bgr: np.ndarray, view: str, card_side: str = "any") -> np.ndarray:
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
    for k in blur_ks:
        gray = cv2.GaussianBlur(base, (k, k), 0)
        for t1, t2 in ((40, 120), (30, 90), (25, 80)):
            contours = _contours_from_canny(gray, t1, t2)
            cands = [
                _largest_quad_from_contours(contours, bgr.shape, min_area_ratio=0.03),
                _largest_id1_rect_from_contours(contours, bgr.shape, min_area_ratio=0.03),
            ]
            cands.extend(_bright_blob_quads(gray, bgr.shape, min_area_ratio=0.03))
            for cand in cands:
                if cand is None:
                    continue
                sc = _quad_score(
                    cand,
                    cv2.contourArea(cand.astype(np.float32)),
                    img_area,
                    bgr.shape,
                    card_side,
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
        quad = _largest_quad_from_contours(contours, bgr.shape, min_area_ratio=0.02)
        if quad is None:
            quad = _largest_id1_rect_from_contours(contours, bgr.shape, min_area_ratio=0.02)

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


def compute_card_geometry(bgr: np.ndarray, view: str, settings: Settings | None = None) -> CardGeometry:
    """
    카드(메트릭 앵커) 검출 + σ 산출.

    v2에서 카드는 **선택적 앵커**다. 검출 실패 시 날조된 사각형으로 진행하는 대신
    깊이추정 단독 경로(+ tier 캡)로 넘어가는 것이 원칙이다(`archimedes-v2-single-photo.mdc` §3.2).
    `allow_card_fallback` 은 데모 전용 완화 스위치이며 기본 OFF.
    """
    try:
        quad = detect_card_quad(bgr, view, _card_side_from(settings))
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
    bgr: np.ndarray, view: str, settings: Settings | None = None
) -> CardGeometry | None:
    """앵커가 선택적인 v2 단일사진 경로용 — 검출 실패를 예외 대신 None 으로."""
    try:
        return compute_card_geometry(bgr, view, settings)
    except PipelineError:
        return None
