"""
스케일 융합 — v2 의 핵심 (`archimedes-v2-single-photo.mdc` §3).

단안 깊이 모델 출력 D̂ 와 참 깊이 D 를 아핀으로 두고 **카드(메트릭 앵커)** 위에서
(s, t) 를 최소제곱으로 푼다.

    D(p) = s · D̂(p) + t

카드는 실측 치수(ISO/IEC 7810 ID-1)를 알기 때문에 PnP 로 평면 포즈를 얻으면
카드 영역 **모든 픽셀의 참 깊이**를 해석적으로 계산할 수 있다. 즉 앵커는
스케일 보정기이면서 동시에 **per-job 깊이 정확도 계측기**다(§7.1).

왜 필요한가: 접사 소물체에서 metric depth 는 절대 스케일이 흔들리는데,
크기 오차는 부피에서 3제곱으로 증폭된다(거리 10% → 무게 33%).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.constants import ID1_HEIGHT_MM, ID1_WIDTH_MM
from app.pipeline.backends.types import DepthKind, DepthMap
from app.pipeline.camera import Intrinsics, pixel_rays
from app.pipeline.card import CardGeometry, card_edge_lengths_px
from app.pipeline.exceptions import PipelineError
from app.pipeline.reconstruct import SupportPlane

log = logging.getLogger(__name__)

# 카드 영역 깊이 분산이 이 비율보다 작으면 s 가 사실상 식별되지 않는다
# (정면에 가까운 카드·상수 깊이 스텁). 그때는 s=1 로 두고 오프셋만 맞춘다.
_ILL_CONDITION_CV = 1e-3
# 융합에 쓰지 않고 남겨 두는 홀드아웃 비율 — 과적합 없는 RMSE 를 위해
_HOLDOUT_STRIDE = 2


@dataclass
class ScaleFusionResult:
    depth_mm: np.ndarray
    scale_s: float
    shift_t_mm: float
    method: str  # anchor_affine | anchor_offset_only | metric_passthrough
    anchor_used: bool
    ill_conditioned: bool
    depth_rmse_mm: float | None = None
    depth_mae_mm: float | None = None
    n_fit_px: int = 0
    n_holdout_px: int = 0
    card_distance_mm: float | None = None
    # 물체가 놓인 바닥면 — 역투영 부피(height_field)가 이걸 쓴다
    support_plane: SupportPlane | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_meta(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "anchor_used": self.anchor_used,
            "ill_conditioned": self.ill_conditioned,
            "scale_s": round(self.scale_s, 6),
            "shift_t_mm": round(self.shift_t_mm, 4),
            "card_distance_mm": (
                round(self.card_distance_mm, 2) if self.card_distance_mm is not None else None
            ),
            # §7.1 평가표 "거리정확도" — 홀드아웃 기준이라 과적합이 없다
            "depth_rmse_mm": round(self.depth_rmse_mm, 3) if self.depth_rmse_mm else None,
            "depth_mae_mm": round(self.depth_mae_mm, 3) if self.depth_mae_mm else None,
            "n_fit_px": self.n_fit_px,
            "n_holdout_px": self.n_holdout_px,
            **self.meta,
        }


def card_object_points_mm(quad_px: np.ndarray) -> np.ndarray:
    """
    카드 평면 좌표계(원점=카드 중심, mm)에서의 4코너 — 이미지 쿼드 순서와 대응.

    `order_quad_points` 는 TL,TR,BR,BL 순이지만 카드가 세로로 누우면 TL→TR 이
    **짧은 변**일 수 있다. 실제 변 길이로 어느 쪽이 85.60mm 축인지 판별한다.
    """
    q = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
    e_tl_tr = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    e_tr_br = 0.5 * (np.linalg.norm(q[2] - q[1]) + np.linalg.norm(q[3] - q[0]))
    if e_tl_tr >= e_tr_br:
        hx, hy = ID1_WIDTH_MM / 2.0, ID1_HEIGHT_MM / 2.0
    else:
        hx, hy = ID1_HEIGHT_MM / 2.0, ID1_WIDTH_MM / 2.0
    return np.array(
        [[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]],
        dtype=np.float64,
    )


def solve_card_plane(card: CardGeometry, K: Intrinsics) -> tuple[np.ndarray, float]:
    """
    카드 평면 PnP → 카메라 좌표계 평면 (n, d), 즉 `n · X = d`.

    Returns (normal_unit, d_mm). d 는 카메라 원점에서 평면까지의 부호 있는 거리.
    """
    obj = card_object_points_mm(card.quad_px)
    img = np.asarray(card.quad_px, dtype=np.float64).reshape(4, 1, 2)
    Kmat = K.matrix()
    dist = np.zeros((4, 1), dtype=np.float64)

    rvec = tvec = None
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rv, tv = cv2.solvePnP(obj, img, Kmat, dist, flags=flag)
        except cv2.error:
            continue
        if ok:
            rvec, tvec = rv, tv
            break
    if rvec is None or tvec is None:
        raise PipelineError(
            "ERR_ANCHOR_AMBIGUOUS",
            "Card plane pose could not be solved from the quad",
            error_severity="soft",
            suggested_action="retake_photo",
        )

    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2].astype(np.float64)  # 카드 평면 법선(카메라 좌표계)
    t = tvec.reshape(3).astype(np.float64)
    d = float(n @ t)
    if d < 0:  # 카메라 뒤에 있는 해 → 법선 뒤집기
        n, d = -n, -d
    if not np.isfinite(d) or d <= 1e-6:
        raise PipelineError("ERR_ANCHOR_AMBIGUOUS", f"Degenerate card plane distance: {d}")
    return n, d


def plane_depth_map(
    n: np.ndarray, d: float, K: Intrinsics, ys: np.ndarray, xs: np.ndarray
) -> np.ndarray:
    """주어진 픽셀들에서 평면까지의 **참 깊이 Z**(mm)."""
    rx, ry = pixel_rays(K, xs.astype(np.float64), ys.astype(np.float64))
    denom = n[0] * rx + n[1] * ry + n[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = d / denom
    return z


def card_interior_pixels(
    card: CardGeometry, shape: tuple[int, int], erode_frac: float = 0.06
) -> tuple[np.ndarray, np.ndarray]:
    """
    카드 내부 픽셀 좌표 (ys, xs).

    테두리는 검출 오차·그림자가 몰리므로 안쪽으로 살짝 침식해서 쓴다.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(card.quad_px, dtype=np.int32)], 255)
    long_px, _ = card_edge_lengths_px(card.quad_px)
    k = max(3, round(long_px * erode_frac) | 1)
    eroded = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    if int((eroded > 0).sum()) < 64:
        eroded = mask
    ys, xs = np.where(eroded > 0)
    return ys, xs


def fuse_scale(
    depth: DepthMap,
    K: Intrinsics,
    card: CardGeometry | None,
    *,
    require_anchor: bool = False,
) -> ScaleFusionResult:
    """
    깊이맵을 **mm 단위 절대 깊이**로 만든다.

    - 앵커 있음 → 카드 평면 참 깊이에 아핀 정합 (s, t)
    - 앵커 없음 + METRIC → 그대로 통과, 신뢰도는 상위에서 캡
    - 앵커 없음 + 스케일 미정 → `ERR_SCALE_UNRESOLVED`
    """
    if card is None or card.used_fallback_quad:
        # 폴백 쿼드는 우리가 지어낸 사각형이라 앵커로 쓸 수 없다
        reason = "no card detected" if card is None else "card quad was fabricated (fallback)"
        if require_anchor:
            raise PipelineError(
                "ERR_SCALE_UNRESOLVED",
                f"Anchor required but unavailable ({reason}). "
                "신용카드를 귀금속과 같은 바닥에 함께 두고 다시 촬영해 주세요.",
                error_severity="soft",
                suggested_action="retake_photo",
            )
        if depth.kind is not DepthKind.METRIC:
            raise PipelineError(
                "ERR_SCALE_UNRESOLVED",
                f"Depth backend is {depth.kind.value} (scale undetermined) and no anchor "
                f"is available ({reason}). 신용카드를 함께 놓고 촬영하거나 metric depth "
                f"백엔드를 사용해 주세요.",
                error_severity="soft",
                suggested_action="retake_photo",
            )
        return ScaleFusionResult(
            depth_mm=depth.depth.astype(np.float32),
            scale_s=1.0,
            shift_t_mm=0.0,
            method="metric_passthrough",
            anchor_used=False,
            ill_conditioned=False,
            meta={"reason": reason},
        )

    n, d = solve_card_plane(card, K)
    ys, xs = card_interior_pixels(card, depth.depth.shape)
    if ys.size < 32:
        raise PipelineError("ERR_ANCHOR_AMBIGUOUS", "Card interior too small to fit scale")

    d_true = plane_depth_map(n, d, K, ys, xs)
    d_hat = depth.depth[ys, xs].astype(np.float64)
    ok = np.isfinite(d_true) & np.isfinite(d_hat) & (d_true > 0)
    if depth.valid is not None:
        ok &= depth.valid[ys, xs]
    if int(np.count_nonzero(ok)) < 32:
        raise PipelineError("ERR_DEPTH_FAILED", "Not enough valid depth samples on the card")

    d_true, d_hat = d_true[ok], d_hat[ok]
    # 홀드아웃은 격자 간격으로 나눠 공간적으로 골고루 섞이게 한다
    idx = np.arange(d_true.size)
    fit = idx % _HOLDOUT_STRIDE != 0
    hold = ~fit
    if fit.sum() < 16 or hold.sum() < 16:
        fit = np.ones_like(idx, dtype=bool)
        hold = np.zeros_like(idx, dtype=bool)

    s, t, ill = _fit_affine(d_hat[fit], d_true[fit])

    depth_mm = (s * depth.depth.astype(np.float64) + t).astype(np.float32)

    rmse = mae = None
    if hold.any():
        pred = s * d_hat[hold] + t
        err = pred - d_true[hold]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))

    return ScaleFusionResult(
        depth_mm=depth_mm,
        scale_s=float(s),
        shift_t_mm=float(t),
        method="anchor_offset_only" if ill else "anchor_affine",
        anchor_used=True,
        ill_conditioned=ill,
        depth_rmse_mm=rmse,
        depth_mae_mm=mae,
        n_fit_px=int(fit.sum()),
        n_holdout_px=int(hold.sum()),
        card_distance_mm=float(d),
        support_plane=SupportPlane(normal=n, d_mm=float(d)),
        meta={"depth_kind": depth.kind.value, "K_source": K.source},
    )


def _fit_affine(d_hat: np.ndarray, d_true: np.ndarray) -> tuple[float, float, bool]:
    """
    최소제곱 `d_true ≈ s·d_hat + t`.

    카드가 정면에 가까우면 참 깊이가 거의 상수라 s 가 **식별되지 않는다**
    (상수 깊이 스텁도 마찬가지). 그때 억지로 회귀하면 s 가 폭주하므로,
    s=1 로 고정하고 오프셋만 맞춘 뒤 `ill_conditioned` 로 알린다.
    """
    mean_hat = float(d_hat.mean())
    spread_hat = float(d_hat.std())
    spread_true = float(d_true.std())
    scale_ref = max(abs(mean_hat), 1e-9)

    degenerate = (
        spread_hat / scale_ref < _ILL_CONDITION_CV
        or spread_true / max(abs(float(d_true.mean())), 1e-9) < _ILL_CONDITION_CV
    )
    if not degenerate:
        var = float(((d_hat - mean_hat) ** 2).mean())
        cov = float(((d_hat - mean_hat) * (d_true - d_true.mean())).mean())
        s = cov / var if var > 0 else 0.0
        if np.isfinite(s) and s > 0:
            return s, float(d_true.mean() - s * mean_hat), False
        log.info("affine scale fit rejected (s=%s); falling back to offset-only", s)

    return 1.0, float(d_true.mean() - mean_hat), True
