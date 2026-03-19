from __future__ import annotations

from io import BytesIO

from PIL import Image

from app.config import Settings
from app.constants import VIEW_ORDER
from app.pipeline.exceptions import PipelineError
from app.s3util import download_object


def load_views_from_s3(settings: Settings, keys: dict[str, str]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for v in VIEW_ORDER:
        if v not in keys:
            raise PipelineError("ERR_VIEWS", f"Missing view key: {v}", retry_step=v)
        try:
            out[v] = download_object(settings, keys[v])
        except Exception as e:  # noqa: BLE001
            raise PipelineError(
                "ERR_DOWNLOAD", f"Failed to download {v}: {e}", retry_step=v
            ) from e
    return out


def bytes_to_bgr(image_bytes: bytes) -> tuple:
    import numpy as np
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise PipelineError("ERR_DECODE", "Could not decode image", retry_step=None)
    return bgr


def collect_exif(image_bytes: bytes) -> dict:
    """Minimal EXIF for Precision/logging (Phase 1)."""
    meta: dict = {}
    try:
        im = Image.open(BytesIO(image_bytes))
        meta["format"] = im.format
        exif = im.getexif()
        if exif and 274 in exif:
            meta["orientation"] = exif[274]
    except Exception:  # noqa: BLE001
        pass
    return meta
