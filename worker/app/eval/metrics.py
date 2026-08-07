"""
평가 지표 (연구개발계획서 「평가방법 및 환경」 3·4번).

계획서가 지정한 절차를 그대로 따른다:
groundtruth 와 예측의 차이를 구해 제곱·합산·표본수로 나눈 뒤 루트 → **RMSE**.
낮을수록 좋다.

두 축을 잰다.
- **거리 정확도**: 앵커(카드) 실측 치수로 계산한 참 깊이 vs 깊이 모델 추정.
  카드가 프레임에 있으면 별도 groundtruth 수집 없이 **매 job** 측정된다
  (`scale_fusion` 홀드아웃 RMSE).
- **중량 정확도**: 저울 실측 vs 추정 (`mass_feedback` 테이블).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorStats:
    n: int
    rmse: float
    mae: float
    mape_pct: float | None
    bias: float  # 평균 부호 오차 — 계통 편향(과대/과소)을 본다

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "mape_pct": round(self.mape_pct, 3) if self.mape_pct is not None else None,
            "bias": round(self.bias, 4),
        }


def error_stats(predicted: list[float], actual: list[float]) -> ErrorStats:
    """
    RMSE / MAE / MAPE / bias.

    MAPE 는 실측이 0 인 표본을 제외하고 계산한다(0 나눗셈 방지).
    표본이 없으면 `n=0` 에 0 을 채워 돌려준다 — 호출부가 분기 없이 집계할 수 있게.
    """
    if len(predicted) != len(actual):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(actual)}")
    n = len(predicted)
    if n == 0:
        return ErrorStats(0, 0.0, 0.0, None, 0.0)

    errs = [p - a for p, a in zip(predicted, actual)]
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    mae = sum(abs(e) for e in errs) / n
    bias = sum(errs) / n

    pct = [abs(e) / abs(a) for e, a in zip(errs, actual) if a != 0]
    mape = 100.0 * sum(pct) / len(pct) if pct else None
    return ErrorStats(n=n, rmse=rmse, mae=mae, mape_pct=mape, bias=bias)


def group_error_stats(
    rows: list[tuple[str, float, float]],
) -> dict[str, ErrorStats]:
    """
    `(그룹키, 추정, 실측)` 목록 → 그룹별 지표 + `"__all__"` 전체.

    제품 형태별로 나눠 봐야 어느 카테고리의 α 가 어긋났는지 보인다.
    """
    buckets: dict[str, tuple[list[float], list[float]]] = {}
    for key, pred, act in rows:
        p, a = buckets.setdefault(key, ([], []))
        p.append(pred)
        a.append(act)

    out = {k: error_stats(p, a) for k, (p, a) in buckets.items()}
    out["__all__"] = error_stats([r[1] for r in rows], [r[2] for r in rows])
    return out
