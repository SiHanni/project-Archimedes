from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.constants import ID1_HEIGHT_MM, ID1_WIDTH_MM
from app.pipeline.exceptions import PipelineError
from app.pipeline.precision_homography import evaluate_card_homography_precision


@dataclass
class CardGeometry:
    sigma_mm_per_px: float
    quad_px: np.ndarray  # 4x2 float32 order TL,TR,BR,BL
    warped_preview: np.ndarray | None = None
    homography_3x3: np.ndarray | None = None  # getPerspectiveTransform (src image → card canonical)
    precision_pose_candidate: bool = False
    precision_solution_count: int = 0


def _largest_quad_from_contours(
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
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            if area > best_area:
                best_area = area
                best = approx.reshape(4, 2).astype(np.float32)
    return best


def detect_card_quad(bgr: np.ndarray, view: str) -> np.ndarray:
    """Find credit-card-like quadrilateral in image (pixels)."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:25]
    quad = _largest_quad_from_contours(contours, bgr.shape, min_area_ratio=0.03)
    if quad is None:
        # Fallback: adaptive threshold for high-contrast card on desk
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5
        )
        contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:25]
        quad = _largest_quad_from_contours(contours, bgr.shape, min_area_ratio=0.02)
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


def compute_card_geometry(bgr: np.ndarray, view: str) -> CardGeometry:
    quad = detect_card_quad(bgr, view)
    return warp_card_and_sigma(bgr, quad)
