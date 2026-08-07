"""
뷰 ↔ 월드 축 매핑 **단일 소스**.

`geometry_g1`(이미지→월드)과 `geometry_project`(월드→이미지)가 각자 if-체인으로
축을 매핑하던 것을 한 테이블로 합친다. 두 방향이 같은 표를 보므로 어긋날 수 없다.

## 월드 좌표계 (project-concept §3.2)
- 바닥(촬영대) = X–Y 평면, **+Z 가 위**(중력 반대).
- 정면 뷰의 "앞"이 **+Y**.

## 촬영 규약 C1 (반드시 UX 로 강제할 것)
5뷰 모드에서 각 컷은 아래 자세로 촬영한다고 **가정**한다. 이 규약이 깨지면
축 부호가 뒤집혀 슬랩 교집합이 무너진다(→ `ERR_VOLUME`).

| 스텝 | 카메라 위치 → 시선 | 카메라 up | image-right | image-up |
|------|--------------------|-----------|-------------|----------|
| front | +Y 에서 −Y 를 봄   | +Z | −X | +Z |
| back  | −Y 에서 +Y 를 봄   | +Z | +X | +Z |
| left  | +X 에서 −X 를 봄   | +Z | +Y | +Z |
| right | −X 에서 +X 를 봄   | +Z | −Y | +Z |
| top   | +Z 에서 −Z 를 봄   | +Y | +X | +Y |

`image-right = forward × up` (오른손 좌표계)로 유도된다.

## 이전 구현의 결함
`left`/`right` 가 `u~z, v~y` 로 되어 있었다. 바닥에 놓인 물체를 옆에서 찍으면
**이미지 세로축이 높이(z)** 이므로 `u~y, v~z` 가 맞다. 이 스왑 때문에 z 구간이
"정면의 높이 ∩ 좌측의 폭"으로 교차돼, 물체가 카드 옆에 있을 때(§4 프로토콜)
교집합이 비거나 엉뚱한 값이 나왔다. 상세: `archimedes-v2-single-photo.mdc` §0.4 #2.
"""

from __future__ import annotations

from typing import Literal

Axis = Literal["x", "y", "z"]

# view -> (u축, u부호, v축, v부호)
#   u = 이미지 가로(오른쪽 +), v = 이미지 세로(위쪽 +, 이미지 y 는 아래로 증가하므로 부호 반전 후)
VIEW_AXIS_MAP: dict[str, tuple[Axis, float, Axis, float]] = {
    "front": ("x", -1.0, "z", +1.0),
    "back": ("x", +1.0, "z", +1.0),
    "left": ("y", +1.0, "z", +1.0),
    "right": ("y", -1.0, "z", +1.0),
    "top": ("x", +1.0, "y", +1.0),
}

_DEFAULT = VIEW_AXIS_MAP["front"]


def axes_for_view(view: str) -> tuple[Axis, float, Axis, float]:
    return VIEW_AXIS_MAP.get(view, _DEFAULT)


def signed_interval(lo: float, hi: float, sign: float) -> tuple[float, float]:
    """부호를 적용해 (lo, hi) 순서를 유지한 구간으로 되돌린다."""
    a, b = lo * sign, hi * sign
    return (a, b) if a <= b else (b, a)


def view_world_intervals(
    view: str, bbox_uv_mm: tuple[float, float, float, float]
) -> dict[Axis, tuple[float, float]]:
    """
    뷰의 (u_min, u_max, v_min, v_max) mm 실루엣 bbox → 이 뷰가 제약하는 **월드 축 구간**.

    한 뷰는 2개 축만 제약한다(시선 축은 제약하지 못함) — 그게 Visual Hull 의 본질이다.
    """
    u_axis, u_sign, v_axis, v_sign = axes_for_view(view)
    u0, u1, v0, v1 = bbox_uv_mm
    return {
        u_axis: signed_interval(u0, u1, u_sign),
        v_axis: signed_interval(v0, v1, v_sign),
    }
