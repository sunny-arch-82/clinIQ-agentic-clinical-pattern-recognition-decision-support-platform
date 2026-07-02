"""Shared decision, evidence, pattern, and workflow models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecommendationType(str, Enum):
    LIKELY_SUPPORTED = "Likely Supported"
    REQUIRES_MANUAL_REVIEW = "Requires Manual Review"
    ADDITIONAL_DOCUMENTATION = "Additional Documentation Recommended"
    EVIDENCE_INSUFFICIENT = "Evidence Insufficient"
    LIKELY_NOT_SUPPORTED = "Likely Not Supported"


class ConfidenceLevel(str, Enum):
    NOT_DISPLAYED = "Not Displayed"


class PatternFinding(BaseModel):
    category: str = ""
    severity: str = "info"
    message: str = ""
    related_fields: list[str] = Field(default_factory=list)


class PatternAnalysis(BaseModel):
    summary: str = ""
    relationships: list[str] = Field(default_factory=list)
    clinical_patterns: list[str] = Field(default_factory=list)
    alignment_checks: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    inconsistencies: list[PatternFinding] = Field(default_factory=list)

    def reviewer_points(self) -> list[str]:
        points: list[str] = []

        points.extend(self.relationships[:3])
        points.extend(self.clinical_patterns[:4])
        points.extend(self.alignment_checks[:3])

        if self.missing_information:
            points.extend(self.missing_information[:3])

        if not points and self.summary:
            points.append(self.summary)

        return points


class EvidenceItem(BaseModel):
    collection: str
    source: str
    content: str
    retrieved_for: str = ""
    url: str | None = None
    internal_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    clinical: list[EvidenceItem] = Field(default_factory=list)
    policies: list[EvidenceItem] = Field(default_factory=list)
    coding: list[EvidenceItem] = Field(default_factory=list)
    medical_necessity: list[EvidenceItem] = Field(default_factory=list)
    historical: list[EvidenceItem] = Field(default_factory=list)

    def all_items(self) -> list[EvidenceItem]:
        return [
            *self.clinical,
            *self.policies,
            *self.coding,
            *self.medical_necessity,
            *self.historical,
        ]


class VerificationCheck(BaseModel):
    name: str
    status: str
    finding: str
    reviewer_action: str


class VerificationResult(BaseModel):
    overall_status: str = "Internal Verification Required"
    summary: str = ""
    checks: list[VerificationCheck] = Field(default_factory=list)

    def reviewer_points(self) -> list[str]:
        return [
            f"{check.name}: {check.status} — {check.reviewer_action}"
            for check in self.checks
        ]


class DecisionRecommendation(BaseModel):
    recommendation: RecommendationType
    reasoning_summary: str
    decision_support: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    suggested_next_actions: list[str] = Field(default_factory=list)
    reviewer_notes: str = ""

    clinical_evidence_position: str = "Not Assessed"
    verification_status: str = "Internal Verification Required"
    final_recommendation_basis: str = ""

    # Kept only for backward compatibility if Streamlit still references it.
    confidence: ConfidenceLevel = ConfidenceLevel.NOT_DISPLAYED


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    case_id: str | None = None
    raw_case: dict[str, Any] | None = None
    normalized_case: dict[str, Any] | None = None

    pattern_analysis: PatternAnalysis | None = None
    query_plan: dict[str, Any] | None = None
    evidence: RetrievedEvidence | None = None
    evidence_review: dict[str, Any] | None = None
    verification_result: VerificationResult | None = None
    decision: DecisionRecommendation | None = None

    report_path: str | None = None
    retrieval_attempts: int = 0
    workflow_steps: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    override_model: str | None = None