"""
세그멘테이션 산출물 시각화·추출 (연구개발계획서 Step 1 — 세미-오토 라벨링).

세 가지를 만든다.

- **overlay**: 원본 위에 귀금속 외곽선·반투명 채움 + 카드 외곽선.
  사람이 "제대로 땄는지" 한눈에 검수하는 용도(평가·디버깅).
- **mask**: 이진 마스크 원본 해상도. 학습·라벨 저장용.
- **cutout**: 배경을 지운 RGBA 누끼.

그리고 **폴리곤 좌표**를 결과 meta 에 실어 준다. 이미지 파일 없이도 라벨을
그대로 쓸 수 있어야 데이터셋으로 굴릴 수 있다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

# 검수용 이미지는 원본 해상도가 필요 없다 — 전송·저장 비용만 커진다
_PREVIEW_MAX_SIDE = 1600
# 폴리곤 단순화 강도 (둘레 대비). 라벨로 쓸 만큼은 남기고 점 수는 줄인다
_POLY_EPS_RATIO = 0.002
_JEWEL_COLOR = (60, 220, 255)  # BGR — 금색 계열
_CARD_COLOR = (255, 170, 60)


@dataclass
class SegmentationAssets:
    overlay_jpg: bytes
    mask_png: bytes
    cutout_png: bytes
    polygon: list[list[int]]
    image_size: tuple[int, int]  # (width, height)

    def as_meta(self) -> dict[str, Any]:
        return {
            "polygon_xy": self.polygon,
            "polygon_points": len(self.polygon),
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
        }


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_polygon(mask: np.ndarray) -> list[list[int]]:
    """오토 라벨링용 폴리곤 (픽셀 좌표). 이미지 없이도 라벨로 쓸 수 있다."""
    cnt = largest_contour(mask)
    if cnt is None:
        return []
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, _POLY_EPS_RATIO * peri, True)
    return [[int(p[0][0]), int(p[0][1])] for p in approx]


def _downscale(img: np.ndarray, max_side: int = _PREVIEW_MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    s = max_side / float(longest)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def build_assets(
    bgr: np.ndarray,
    jewel_mask: np.ndarray,
    card_quad: np.ndarray | None = None,
) -> SegmentationAssets:
    """원본 + 마스크 → 검수용 오버레이 · 라벨용 마스크 · 누끼."""
    h, w = bgr.shape[:2]
    binary = ((jewel_mask > 0).astype(np.uint8)) * 255

    # ── overlay: 반투명 채움 + 외곽선 ──
    overlay = bgr.copy()
    tint = np.zeros_like(bgr)
    tint[:] = _JEWEL_COLOR
    sel = binary > 0
    overlay[sel] = cv2.addWeighted(bgr, 0.55, tint, 0.45, 0)[sel]

    thickness = max(2, round(max(h, w) / 500))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, _JEWEL_COLOR, thickness)
    if card_quad is not None:
        cv2.polylines(
            overlay,
            [np.asarray(card_quad, dtype=np.int32)],
            True,
            _CARD_COLOR,
            thickness,
        )

    ok_overlay, overlay_buf = cv2.imencode(
        ".jpg", _downscale(overlay), [int(cv2.IMWRITE_JPEG_QUALITY), 88]
    )
    # 마스크는 라벨 원본이므로 **해상도를 줄이지 않는다**
    ok_mask, mask_buf = cv2.imencode(".png", binary)

    # ── cutout: 배경 투명 ──
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = binary
    ys, xs = np.where(sel)
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        rgba = rgba[y0 : y1 + 1, x0 : x1 + 1]
    ok_cut, cut_buf = cv2.imencode(".png", _downscale(rgba))

    if not (ok_overlay and ok_mask and ok_cut):
        raise RuntimeError("failed to encode segmentation assets")

    return SegmentationAssets(
        overlay_jpg=overlay_buf.tobytes(),
        mask_png=mask_buf.tobytes(),
        cutout_png=cut_buf.tobytes(),
        polygon=contour_polygon(binary),
        image_size=(w, h),
    )
