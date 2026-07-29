"""
G1: 뷰별 실루엣 bbox 를 카드 중심 기준 (u,v) mm 평면 좌표로 옮긴다.

축 부호·월드 매핑은 `view_axes.VIEW_AXIS_MAP` **단일 소스**를 따른다.
이 모듈은 부호 없는 (u,v) mm 만 만들고, 월드 축 할당은
`view_axes.view_world_intervals` 가 담당한다.

이전 구현은 여기서 뷰별 미러를 직접 넣었고 `geometry_project` 는 별도 if-체인을
갖고 있어 두 방향이 어긋났다 — 상세: `archimedes-v2-single-photo.mdc` §0.4 #2.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.card import CardGeometry


def jewel_bbox_uv_mm(
    mask: np.ndarray, card: CardGeometry, view: str | None = None
) -> tuple[float, float, float, float]:
    """
    Returns (u_min, u_max, v_min, v_max) in mm — 카드 중심 원점,
    u = 이미지 가로(+오른쪽), v = 이미지 세로(+위쪽).

    `view` 는 하위 호환용 인자이며 더 이상 부호에 관여하지 않는다.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return 0.0, 0.0, 0.0, 0.0
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    ccx = float(card.quad_px[:, 0].mean())
    ccy = float(card.quad_px[:, 1].mean())
    s = card.sigma_mm_per_px

    u0 = (x0 - ccx) * s
    u1 = (x1 - ccx) * s
    # 이미지 y 는 아래로 증가 → 위쪽이 + 가 되도록 반전
    v0 = -(y1 - ccy) * s
    v1 = -(y0 - ccy) * s
    return min(u0, u1), max(u0, u1), min(v0, v1), max(v0, v1)
