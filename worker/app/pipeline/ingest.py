from __future__ import annotations

import logging
from io import BytesIO

from PIL import Image

from app.config import Settings
from app.constants import VIEW_ORDER
from app.pipeline.exceptions import PipelineError
from app.s3util import download_object

log = logging.getLogger(__name__)

# EXIF 태그
_TAG_ORIENTATION = 274
_TAG_FOCAL_LENGTH = 37386  # FocalLength (mm, rational)
_TAG_FOCAL_LENGTH_35MM = 41989  # FocalLengthIn35mmFilm (mm, int)
_TAG_MAKE = 271
_TAG_MODEL = 272
_EXIF_SUB_IFD = 0x8769


def load_views_from_s3(settings: Settings, keys: dict[str, str]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for v in VIEW_ORDER:
        if v not in keys:
            raise PipelineError("ERR_VIEWS", f"Missing view key: {v}", retry_step=v)
        try:
            out[v] = download_object(settings, keys[v])
        except Exception as e:
            raise PipelineError(
                "ERR_DOWNLOAD", f"Failed to download {v}: {e}", retry_step=v
            ) from e
    return out


def load_images_from_s3(settings: Settings, keys: dict[str, str]) -> dict[str, bytes]:
    """뷰 집합이 가변인 v2 경로용 — `keys` 에 담긴 것만 받는다."""
    out: dict[str, bytes] = {}
    for name, key in keys.items():
        try:
            out[name] = download_object(settings, key)
        except Exception as e:
            raise PipelineError(
                "ERR_DOWNLOAD", f"Failed to download {name}: {e}", retry_step=name
            ) from e
    return out


def apply_exif_orientation(bgr, orientation: int | None):
    """
    EXIF orientation(1~8)을 실제 화소에 적용한다.

    `cv2.imdecode` 는 orientation 을 반영하지 않으므로, 세로로 찍은 폰 사진이
    눕혀진 채로 파이프라인에 들어가 뷰↔월드 축 매핑이 통째로 어긋났다.
    상세: `archimedes-v2-single-photo.mdc` §0.4 #4.

    EXIF 규격의 row0/col0 정의를 따른다:
    1=top/left, 2=top/right, 3=bottom/right, 4=bottom/left,
    5=left/top(transpose), 6=right/top, 7=right/bottom(transverse), 8=left/bottom
    """
    import cv2

    if not orientation or orientation == 1:
        return bgr
    if orientation == 2:
        return cv2.flip(bgr, 1)
    if orientation == 3:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(bgr, 0)
    if orientation == 5:
        return cv2.rotate(cv2.flip(bgr, 0), cv2.ROTATE_90_CLOCKWISE)
    if orientation == 6:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.rotate(cv2.flip(bgr, 1), cv2.ROTATE_90_CLOCKWISE)
    if orientation == 8:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def bytes_to_bgr(image_bytes: bytes, orientation: int | None = None):
    """
    디코드 + **EXIF 회전 적용**.

    `orientation` 을 주지 않으면 바이트에서 직접 읽는다(호출부가 이미 `collect_exif`
    를 했다면 그 값을 넘겨 중복 파싱을 피할 수 있다).
    """
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise PipelineError("ERR_DECODE", "Could not decode image", retry_step=None)
    if orientation is None:
        orientation = read_exif_orientation(image_bytes)
    return apply_exif_orientation(bgr, orientation)


def read_exif_orientation(image_bytes: bytes) -> int | None:
    try:
        exif = Image.open(BytesIO(image_bytes)).getexif()
        if exif:
            val = exif.get(_TAG_ORIENTATION)
            return int(val) if val is not None else None
    except Exception as e:  # noqa: BLE001
        log.debug("EXIF orientation read failed: %s", e)
    return None


def collect_exif(image_bytes: bytes) -> dict:
    """
    orientation + 초점거리 + 기기 모델.

    초점거리는 v2 스케일 융합에서 내부 파라미터 K 를 세우는 데 쓴다
    (`archimedes-v2-single-photo.mdc` §3.3).
    """
    meta: dict = {}
    try:
        im = Image.open(BytesIO(image_bytes))
        meta["format"] = im.format
        meta["width"] = int(im.width)
        meta["height"] = int(im.height)
        exif = im.getexif()
        if not exif:
            return meta
        if exif.get(_TAG_ORIENTATION) is not None:
            meta["orientation"] = int(exif[_TAG_ORIENTATION])
        for tag, key in ((_TAG_MAKE, "make"), (_TAG_MODEL, "model")):
            if exif.get(tag):
                meta[key] = str(exif[tag]).strip()

        # FocalLength 계열은 Exif SubIFD 에 있다
        try:
            sub = dict(exif.get_ifd(_EXIF_SUB_IFD) or {})
        except Exception:  # noqa: BLE001
            sub = {}
        merged = {**dict(exif), **sub}
        if merged.get(_TAG_FOCAL_LENGTH_35MM):
            try:
                meta["focal_length_35mm"] = float(merged[_TAG_FOCAL_LENGTH_35MM])
            except (TypeError, ValueError):
                pass
        if merged.get(_TAG_FOCAL_LENGTH):
            try:
                meta["focal_length_mm"] = float(merged[_TAG_FOCAL_LENGTH])
            except (TypeError, ValueError):
                pass
    except Exception as e:  # noqa: BLE001
        log.debug("EXIF collect failed: %s", e)
    return meta
