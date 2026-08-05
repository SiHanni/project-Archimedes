"""세그 산출물 — 오버레이·마스크·누끼·폴리곤 (계획서 Step 1 세미-오토 라벨링)."""

from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.visualize import build_assets, contour_polygon


def _scene(h: int = 400, w: int = 600) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((h, w, 3), 40, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(img, (100, 120), (260, 250), (30, 200, 240), -1)
    cv2.rectangle(mask, (100, 120), (260, 250), 255, -1)
    return img, mask


def test_polygon_is_usable_as_a_label() -> None:
    """라벨로 쓰려면 이미지 없이도 영역이 복원돼야 한다."""
    _img, mask = _scene()
    poly = contour_polygon(mask)
    assert len(poly) >= 4

    redrawn = np.zeros_like(mask)
    cv2.fillPoly(redrawn, [np.array(poly, dtype=np.int32)], 255)
    inter = int(np.count_nonzero((redrawn > 0) & (mask > 0)))
    union = int(np.count_nonzero((redrawn > 0) | (mask > 0)))
    assert inter / union > 0.97, "폴리곤이 원 마스크를 충실히 재현해야 한다"


def test_polygon_empty_for_empty_mask() -> None:
    assert contour_polygon(np.zeros((50, 50), np.uint8)) == []


def test_assets_are_decodable_images() -> None:
    img, mask = _scene()
    a = build_assets(img, mask, card_quad=np.array([[300, 100], [560, 100], [560, 270], [300, 270]]))

    overlay = cv2.imdecode(np.frombuffer(a.overlay_jpg, np.uint8), cv2.IMREAD_COLOR)
    assert overlay is not None and overlay.ndim == 3

    decoded_mask = cv2.imdecode(np.frombuffer(a.mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded_mask is not None
    # 마스크는 라벨 원본이므로 **해상도를 줄이면 안 된다**
    assert decoded_mask.shape == mask.shape
    assert set(np.unique(decoded_mask)).issubset({0, 255})

    cutout = cv2.imdecode(np.frombuffer(a.cutout_png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert cutout is not None and cutout.shape[2] == 4, "누끼는 알파 채널이 있어야 한다"


def test_cutout_is_cropped_to_the_object() -> None:
    """누끼는 물체 바운딩박스로 잘라 낸다 — 배경 여백을 들고 다닐 이유가 없다."""
    img, mask = _scene()
    a = build_assets(img, mask)
    cutout = cv2.imdecode(np.frombuffer(a.cutout_png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert cutout.shape[0] < img.shape[0]
    assert cutout.shape[1] < img.shape[1]
    # 잘라 낸 영역 안은 대부분 불투명해야 한다(사각형 물체라 거의 전부)
    assert float((cutout[:, :, 3] > 0).mean()) > 0.95


def test_overlay_marks_the_object() -> None:
    """오버레이는 사람이 검수하는 용도 — 물체 영역이 원본과 달라져야 보인다."""
    img, mask = _scene()
    a = build_assets(img, mask)
    overlay = cv2.imdecode(np.frombuffer(a.overlay_jpg, np.uint8), cv2.IMREAD_COLOR)
    small_mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
    small_img = cv2.resize(img, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_AREA)
    inside = small_mask > 0
    diff = np.abs(overlay.astype(int) - small_img.astype(int)).mean(axis=2)
    assert diff[inside].mean() > 10, "물체 영역이 시각적으로 표시돼야 한다"


def test_meta_carries_image_size_for_label_scaling() -> None:
    img, mask = _scene()
    meta = build_assets(img, mask).as_meta()
    assert meta["image_width"] == img.shape[1]
    assert meta["image_height"] == img.shape[0]
    assert meta["polygon_points"] == len(meta["polygon_xy"])
