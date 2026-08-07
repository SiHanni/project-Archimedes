"""검출·분할·깊이 백엔드가 주고받는 값 타입."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DepthKind(str, Enum):
    """
    깊이 모델 출력의 **스케일 성격**. 스케일 융합(§3)이 무엇을 풀어야 하는지 결정한다.

    - METRIC: 절대 mm 를 주장 (그래도 접사에서는 드리프트하므로 앵커로 검증한다)
    - AFFINE_INVARIANT: D = s·D̂ + t 의 (s, t) 가 미정 → 앵커 없으면 복원 불가
    - RELATIVE: 순서만 의미 있음(역깊이 등) → 앵커 필수
    """

    METRIC = "metric"
    AFFINE_INVARIANT = "affine_invariant"
    INVERSE_AFFINE = "inverse_affine"
    RELATIVE = "relative"

    @property
    def needs_anchor(self) -> bool:
        return self is not DepthKind.METRIC

    @property
    def affine_in_inverse_depth(self) -> bool:
        """
        출력이 **시차(disparity)** 라서 아핀 관계가 역깊이 공간에서 성립하는가.

        MiDaS·Depth Anything 계열은 `d ≈ α/Z + β` 를 낸다. 이걸 깊이 공간의
        아핀 `Z ≈ s·d + t` 로 맞추면 β 때문에 체계적으로 틀어진다.
        (β=0 일 때만 두 모델이 같아진다)
        """
        return self is DepthKind.INVERSE_AFFINE


@dataclass
class Detection:
    """이미지 좌표계 박스 (x0, y0, x1, y1) — 정수 픽셀, x1/y1 포함."""

    box_xyxy: tuple[int, int, int, int]
    score: float
    label: str  # "jewelry" | "card" | ...

    def area(self) -> int:
        x0, y0, x1, y1 = self.box_xyxy
        return max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)

    def clipped(self, h: int, w: int) -> Detection:
        x0, y0, x1, y1 = self.box_xyxy
        return Detection(
            (
                int(np.clip(x0, 0, w - 1)),
                int(np.clip(y0, 0, h - 1)),
                int(np.clip(x1, 0, w - 1)),
                int(np.clip(y1, 0, h - 1)),
            ),
            self.score,
            self.label,
        )


@dataclass
class SegmentResult:
    """이진 마스크(uint8, 0 또는 255) + 백엔드 메타."""

    mask: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthMap:
    """
    깊이 추정 결과.

    `depth` 단위는 `kind` 에 따른다 — METRIC 이면 mm, 그 외에는 모델 임의 단위.
    스케일 융합을 거친 뒤에는 항상 **mm** 가 된다.
    """

    depth: np.ndarray  # float32 (H, W)
    kind: DepthKind
    valid: np.ndarray | None = None  # bool (H, W); None 이면 전부 유효
    meta: dict[str, Any] = field(default_factory=dict)

    def valid_mask(self) -> np.ndarray:
        if self.valid is not None:
            return self.valid
        return np.isfinite(self.depth)

    def with_depth(self, depth: np.ndarray, kind: DepthKind) -> DepthMap:
        return DepthMap(depth=depth, kind=kind, valid=self.valid, meta=dict(self.meta))
