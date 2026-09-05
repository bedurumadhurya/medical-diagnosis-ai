from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Modality = Literal["chest_xray", "brain_mri"]


class Finding(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    present: bool


class Region(BaseModel):
    label: str
    score: float
    bbox: list[float]
    mask_coverage: float


class AnalyzeResponse(BaseModel):
    request_id: str
    modality: Modality
    model_mode: Literal["trained", "imagenet_demo"]
    findings: list[Finding]
    primary_impression: str
    heatmap_png_b64: str
    overlay_png_b64: str
    segmentation_png_b64: str | None = None
    regions: list[Region] = []
    report: str
    dice_proxy: float | None = None
    disclaimer: str
    latency_ms: float
    agents: dict[str, str] = {}
