from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.constants import VIEW_ORDER

# outline = 에라토스테네스. 기준물 없이 **누끼만** 딴다(크기·무게 없음).
CaptureMode = Literal["single", "multiview", "outline"]


class JobViews(BaseModel):
    """5뷰 모드의 S3 키 묶음."""

    front: str
    top: str
    left: str
    right: str
    back: str


class JobInputRecord(BaseModel):
    """
    MySQL `input_json` 에 저장되는 입력.

    v2 는 **단일 사진**이 기본이고 5뷰는 옵션(고신뢰) 모드다.
    기존 행은 `views` 만 갖고 있으므로 `capture_mode` 를 추론해 하위 호환을 유지한다.
    """

    # v2 단일사진 경로 — S3 key 1개
    image: str | None = None
    # v1/다뷰 경로 — 5뷰 S3 key
    views: JobViews | None = None
    capture_mode: CaptureMode | None = None

    metal: str = "gold"
    purity: str = "18k"
    product_k: str = "ring"
    reference_weight_g: float | None = None
    # 골드바처럼 **깊이로 못 재는 얇은 제품**의 두께(mm). 각인·규격에서 알 수 있다.
    # 주면 관측값 대신 이 값을 쓴다.
    reference_thickness_mm: float | None = None
    # 도금·금박 제품에 **인쇄된 순금 함유량**(g). 부피로는 못 재므로 표기값을 받는다.
    declared_gold_g: float | None = None
    knows_weight: str | None = None  # "yes" | "no" | None — UX only

    @model_validator(mode="after")
    def _resolve_capture_mode(self) -> JobInputRecord:
        if self.capture_mode is None:
            self.capture_mode = "multiview" if self.views else "single"
        if self.capture_mode == "multiview" and self.views is None:
            raise ValueError("capture_mode=multiview requires `views`")
        if self.capture_mode == "single" and not self.image:
            raise ValueError("capture_mode=single requires `image`")
        return self

    def image_keys(self) -> dict[str, str]:
        """다운로드할 {이름: S3 key}. 단일 모드는 `{"front": key}`."""
        if self.capture_mode == "multiview" and self.views is not None:
            dumped = self.views.model_dump()
            return {v: dumped[v] for v in VIEW_ORDER}
        return {"front": self.image or ""}


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
