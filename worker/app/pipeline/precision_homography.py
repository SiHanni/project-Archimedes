"""
카드 호모그래피 분해 — Precision 경로 v0 (실패 시 weak G1만 사용).
OpenCV: decomposeHomographyMat(H, K) + 단순 체리얼리티 휴리스틱.
"""

from __future__ import annotations

import numpy as np

import cv2


def evaluate_card_homography_precision(
    H: np.ndarray,
    image_hw: tuple[int, int],
) -> tuple[bool, int]:
    """
    Returns (pose_candidate_ok, num_solutions).
    K는 초점 미상일 때의 대각 근사(연구/스캐폴드용).
    """
    h, w = image_hw[:2]
    if H is None or H.shape != (3, 3):
        return False, 0
    f = float(max(w, h)) * 1.15
    cx, cy = w * 0.5, h * 0.5
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    try:
        retval, _rotations, _translations, normals = cv2.decomposeHomographyMat(
            H.astype(np.float64), K
        )
    except cv2.error:
        return False, 0
    if retval <= 0:
        return False, 0
    ok_any = False
    for i in range(int(retval)):
        nvec = np.asarray(normals[i]).reshape(-1)
        if nvec.size >= 3 and abs(float(nvec[2])) > 0.25:
            ok_any = True
            break
    return ok_any, int(retval)
