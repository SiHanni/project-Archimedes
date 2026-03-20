from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings
from app.constants import ID1_HEIGHT_MM, ID1_WIDTH_MM
from app.pipeline.exceptions import PipelineError
from app.pipeline.precision_homography import evaluate_card_homography_precision

# ID-1 가로/세로 비 ≈ 1.586 — 실사진에서 왜곡·원근으로 범위를 넓게 둠
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


def _largest_quad_from_contours(
    contours: list,
    shape: tuple[int, int, int] | tuple[int, int],
    min_area_ratio: float,
    eps_factors: tuple[float, ...] = (0.02, 0.035, 0.05),
) -> np.ndarray | None:
    h = shape[0]
    w = shape[1]
    img_area = float(h * w)
    best = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps_f in eps_factors:
            approx = cv2.approxPolyDP(cnt, eps_f * peri, True)
            if len(approx) == 4:
                if area > best_area:
                    best_area = area
                    best = approx.reshape(4, 2).astype(np.float32)
                break
    return best


def _min_area_rect_id1_like(
    cnt: np.ndarray,
    img_area: float,
    min_area_ratio: float,
) -> np.ndarray | None:
    """윤곽이 4각 근사에 실패해도, 최소면적 회전사각형이 카드 비율이면 채택."""
    area = cv2.contourArea(cnt)
    if area < min_area_ratio * img_area:
        return None
    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), _ = rect
    if rw < 1.0 or rh < 1.0:
        return None
    ar = max(rw, rh) / min(rw, rh)
    if not (_ID1_ASPECT_MIN <= ar <= _ID1_ASPECT_MAX):
        return None
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


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
        if quad is None:
            continue
        a = cv2.contourArea(cnt)
        if a > best_area:
            best_area = a
            best = quad
    return best


def _contours_from_canny(gray: np.ndarray, t1: int, t2: int) -> list:
    edges = cv2.Canny(gray, t1, t2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)[:30]


def detect_card_quad(bgr: np.ndarray, view: str) -> np.ndarray:
    """Find credit-card-like quadrilateral in image (pixels)."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    quad = None
    for t1, t2 in ((40, 120), (30, 90), (25, 80)):
        contours = _contours_from_canny(gray, t1, t2)
        quad = _largest_quad_from_contours(contours, bgr.shape, min_area_ratio=0.03)
        if quad is not None:
            break
        quad = _largest_id1_rect_from_contours(contours, bgr.shape, min_area_ratio=0.03)
        if quad is not None:
            break

    if quad is None:
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
            "Card quadrilateral not detected",
            retry_step=view,
        )
    return order_quad_points(quad)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order: top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


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


def warp_card_and_sigma(bgr: np.ndarray, quad: np.ndarray) -> CardGeometry:
    """Warp card to ID-1 aspect; sigma = mm/px along width."""
    dst_w = 856
    dst_h = int(round(dst_w * (ID1_HEIGHT_MM / ID1_WIDTH_MM)))
    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr, H, (dst_w, dst_h))
    sigma = ID1_WIDTH_MM / float(dst_w)
    prec_ok, prec_n = evaluate_card_homography_precision(H, bgr.shape)
    return CardGeometry(
        sigma_mm_per_px=sigma,
        quad_px=quad,
        warped_preview=warped,
        homography_3x3=H,
        precision_pose_candidate=prec_ok,
        precision_solution_count=prec_n,
    )


def compute_card_geometry(bgr: np.ndarray, view: str, settings: Settings | None = None) -> CardGeometry:
    try:
        quad = detect_card_quad(bgr, view)
        geom = warp_card_and_sigma(bgr, quad)
        geom.used_fallback_quad = False
        return geom
    except PipelineError:
        if settings is None or not settings.allow_card_fallback:
            raise
        # 완화 모드: 카드 미검출이어도 임시 사각형으로 파이프라인을 진행(정확도 저하 가능)
        quad = _fallback_quad_from_frame(bgr.shape)
        geom = warp_card_and_sigma(bgr, quad)
        geom.used_fallback_quad = True
        return geom
