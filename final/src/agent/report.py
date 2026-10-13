"""Structured intervention report schema."""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    source: Literal["sql", "model"] = Field(description="where this fact came from")
    fact: str = Field(description="a concrete, citable fact")


class InterventionReport(BaseModel):
    student_id: int
    cutoff_week: int
    p_at_risk: float = Field(ge=0.0, le=1.0)
    risk_band: Literal["low", "moderate", "high"]
    evidence: list[EvidenceItem]
    recommended_action: Literal["monitor", "outreach", "urgent_outreach", "none"]
    action_rationale: str
    uncertainty_caveats: list[str]


class RationaleFill(BaseModel):
    action_rationale: str
    uncertainty_caveats: list[str]
