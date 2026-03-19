from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobViews(BaseModel):
    front: str
    top: str
    left: str
    right: str
    back: str


class JobInputRecord(BaseModel):
    """Stored in MySQL input_json."""

    views: JobViews
    metal: str = "gold"
    purity: str = "18k"
    product_k: str = "ring"
    reference_weight_g: float | None = None
    knows_weight: str | None = None  # "yes" | "no" | None — UX only


class MassRange(BaseModel):
    min_g: float
    estimate_g: float
    max_g: float


class JobResult(BaseModel):
    algorithm_version: str
    V_hull_mm3: float
    V_adj_mm3: float
    mass_est_g: float
    confidence_tier: str
    confidence_pct: float
    mass_range: MassRange | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class PipelineErrorBody(BaseModel):
    code: str
    message: str
    retry_step: str | None = None
