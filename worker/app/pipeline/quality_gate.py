from __future__ import annotations

import cv2
import numpy as np

from app.config import Settings
from app.pipeline.exceptions import PipelineError


def check_image_quality(
    bgr: np.ndarray, settings: Settings, view: str, *, needs_scale: bool = True
) -> None:
    """
    `needs_scale=False` 면 해상도 문턱을 크게 낮춘다.

    해상도 문턱은 **크기를 재기 위한** 것이다. 카드 검출 오차 1px 이 σ 를 통해
    면적에서 제곱, 부피에서 세제곱으로 커지므로, 무게를 내려면 원본급이 필요하다.

    그런데 **외곽선 모드는 크기를 주장하지 않는다.** 그 경로에까지 같은 문턱을
    들이대면, 원본이 없는 사용자가 **모니터 화면을 폰으로 재촬영**하는 우회를
    하게 된다 — 실측: 그렇게 올라온 사진은 모아레 격자 때문에 누끼가 화면 중앙
    밝은 조각만 잡았다. 작은 원본을 그대로 받는 편이 훨씬 낫다.

    **거리 모드도 `needs_scale=False` 로 넘긴다.** 축소는 거리 추정에 영향이 없다 —
    핀홀식이 `Z = f · S / p` 인데 사진을 절반으로 줄이면 `f` 도 `p` 도 함께 절반이
    되어 `Z` 는 불변이다. 크기를 재는 경로처럼 절대 픽셀 수가 곧 mm 가 되는 것이
    아니므로, 같은 문턱을 들이댈 근거가 없다. (초점거리는 EXIF 가 있으면 폭에
    맞춰 환산하고, 없으면 Depth Pro 가 그 이미지 자체의 픽셀 좌표계에서 추정한다)
    """
    h, w = bgr.shape[:2]
    short = min(h, w)
    limit = settings.min_short_edge_px if needs_scale else settings.min_short_edge_outline_px
    if short < limit:
        raise PipelineError(
            "ERR_RESOLUTION",
            f"사진이 너무 작습니다({w}×{h}). 짧은 변이 {limit}px 이상이어야 합니다. "
            + (
                "무게를 재려면 폰 원본 사진이 필요합니다 — 캡처·축소본은 정확도가 "
                "몇 배씩 어긋납니다."
                if needs_scale
                else "물체가 보이는 사진이면 됩니다."
            ),
            retry_step=view,
            error_severity="hard",
            suggested_action="retry_one_view",
        )

    gray = bgr[:, :, 1] if bgr.ndim == 3 else bgr  # green channel quick
    lap = cv2_laplacian_var(gray)
    if lap < settings.blur_laplacian_threshold:
        raise PipelineError(
            "ERR_BLUR",
            f"Laplacian variance {lap:.1f} below threshold",
            retry_step=view,
            error_severity="soft",
            suggested_action="retry_one_view",
        )

    mean = float(gray.mean())
    if mean < 25.0 or mean > 230.0:
        raise PipelineError(
            "ERR_EXPOSURE",
            f"Mean intensity {mean:.1f} out of acceptable range",
            retry_step=view,
            error_severity="soft",
            suggested_action="retry_one_view",
        )


def cv2_laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())
