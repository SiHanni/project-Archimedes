"""
학습형 잔차 보정용 특징 추출 (`archimedes-v2-single-photo.mdc` §4.4).

물리식을 버리지 않고 **잔차만** 학습한다.

    m_final = m_physics · exp(w · x)

로그를 쓰는 이유: 보정은 본질적으로 배수(0.8배·1.2배)이고, 로그 공간에서
선형회귀를 풀면 항상 양수 배수가 나와 무게가 음수가 되는 일이 없다.

특징은 **결과 meta 에 이미 적재된 값만** 쓴다. 재처리 없이 과거 job 으로도
학습셋을 만들 수 있어야 하기 때문이다.
"""

from __future__ import annotations

import math
from typing import Any

# 계수 파일과 순서가 어긋나면 보정이 통째로 틀어지므로 이름을 함께 저장한다.
FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "log_area_mm2",
    "log_aspect",          # 길이/폭 — 가늘고 긴 형태(체인)와 뭉툭한 형태를 가른다
    "log_h_mean_mm",
    "thickness_assumed",   # 두께를 관측 못하고 클램프했는가 (0/1)
    "anchor_used",         # 카드 앵커가 실제로 쓰였는가 (0/1)
    "height_field",        # 부피 모델이 height_field 인가 (0/1)
    "rel_depth_rmse",      # 홀드아웃 거리 RMSE / 카드 거리
)


def _f(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _log1p_pos(v: float) -> float:
    """양수 스케일 특징용. 0 이하는 0 으로 눌러 로그 폭주를 막는다."""
    return math.log1p(v) if v > 0 else 0.0


def extract_features(result: dict[str, Any]) -> list[float] | None:
    """
    job `result_json` → 특징 벡터.

    단일사진(depth) 경로만 대상으로 한다. 다뷰 경로는 부피 산출 방식이 달라
    같은 계수로 보정하면 안 된다. 대상이 아니면 `None`.
    """
    meta = result.get("meta") or {}
    if meta.get("capture_mode") != "single":
        return None

    rec = meta.get("reconstruction") or {}
    fusion = meta.get("scale_fusion") or {}

    area = _f(rec.get("area_proj_mm2"))
    length = _f(rec.get("length_mm"))
    width = _f(rec.get("width_mm"))
    h_mean = _f(rec.get("h_mean_mm"))
    if area <= 0 or h_mean <= 0:
        return None

    aspect = (length / width) if width > 0 else 1.0
    rmse = _f(fusion.get("depth_rmse_mm"))
    dist = _f(fusion.get("card_distance_mm"))
    rel_rmse = (rmse / dist) if dist > 0 else 0.0

    return [
        1.0,
        _log1p_pos(area),
        math.log(max(aspect, 1e-3)),
        _log1p_pos(h_mean),
        1.0 if rec.get("thickness_clamp") else 0.0,
        1.0 if fusion.get("anchor_used") else 0.0,
        1.0 if rec.get("method") == "height_field" else 0.0,
        rel_rmse,
    ]


def target_log_ratio(mass_actual_g: float, mass_est_g: float) -> float | None:
    """학습 타깃 `log(실측/추정)`. 0 이하 값은 제외."""
    if mass_actual_g <= 0 or mass_est_g <= 0:
        return None
    return math.log(mass_actual_g / mass_est_g)


def apply_correction(mass_est_g: float, features: list[float], weights: list[float]) -> float:
    """학습된 계수로 보정. 길이가 안 맞으면 **보정하지 않고 원값**을 돌려준다."""
    if len(features) != len(weights):
        return mass_est_g
    z = sum(f * w for f, w in zip(features, weights))
    # 폭주 방지 — 한 번에 4배 이상 움직이는 보정은 데이터 문제로 본다
    z = max(-1.4, min(1.4, z))
    return mass_est_g * math.exp(z)
