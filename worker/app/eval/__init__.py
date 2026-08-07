"""평가 지표·학습형 잔차 특징 (연구개발계획서 「평가방법 및 환경」·§4.4)."""

from app.eval.features import (
    FEATURE_NAMES,
    apply_correction,
    extract_features,
    target_log_ratio,
)
from app.eval.metrics import ErrorStats, error_stats, group_error_stats

__all__ = [
    "FEATURE_NAMES",
    "ErrorStats",
    "apply_correction",
    "error_stats",
    "extract_features",
    "group_error_stats",
    "target_log_ratio",
]
