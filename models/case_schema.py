"""Pydantic schemas for healthcare case JSON validation and normalization."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SchemaVersion(str, Enum):
    V1 = "1.0"


class CaseInformation(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    schema_version: str = "1.0"
    created_at: str | None = None
    case_type: str | None = "claim_review"


class Patient(BaseModel):
    model_config = ConfigDict(extra="allow")

    patient_id: str
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: str | None = None
    gender: str | None = None
    member_id: str | None = None


class Provider(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider_id: str | None = None
    name: str | None = None
    npi: str | None = None
    specialty: str | None = None


class Payer(BaseModel):
    model_config = ConfigDict(extra="allow")

    payer_id: str | None = None
    name: str | None = None
    plan_type: str | None = None
    policy_id: str | None = None


class Encounter(BaseModel):
    model_config = ConfigDict(extra="allow")

    encounter_id: str | None = None
    encounter_type: str | None = None
    admission_date: str | None = None
    discharge_date: str | None = None
    place_of_service: str | None = None


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str
    system: str = "ICD-10-CM"
    description: str | None = None
    rank: int | None = None
    is_primary: bool = False


class Procedure(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str
    system: str = "CPT"
    description: str | None = None
    date_of_service: str | None = None
    units: int = 1
    charge_amount: float | None = None


class Medication(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    ndc: str | None = None
    dosage: str | None = None
    frequency: str | None = None


class LabResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    test_name: str
    result_value: str | None = None
    reference_range: str | None = None
    units: str | None = None
    date: str | None = None


class ClaimLine(BaseModel):
    model_config = ConfigDict(extra="allow")

    line_number: int | None = None
    procedure_code: str | None = None
    diagnosis_pointers: list[int] = Field(default_factory=list)
    billed_amount: float | None = None
    allowed_amount: float | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="allow")

    claim_id: str | None = None
    claim_type: str | None = None
    total_billed: float | None = None
    service_from: str | None = None
    service_to: str | None = None
    lines: list[ClaimLine] = Field(default_factory=list)


class SupportingDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str | None = None
    document_type: str
    title: str | None = None
    summary: str | None = None
    date: str | None = None


class HealthcareCase(BaseModel):
    """Root schema for standardized healthcare claim cases."""

    model_config = ConfigDict(extra="allow")

    case_information: CaseInformation
    patient: Patient
    provider: Provider | None = None
    payer: Payer | None = None
    encounter: Encounter | None = None
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    claim: Claim | None = None
    supporting_documents: list[SupportingDocument] = Field(default_factory=list)

    @field_validator("diagnoses", mode="before")
    @classmethod
    def ensure_diagnoses_list(cls, value: Any) -> list[Any]:
        return value or []

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a normalized dict safe for agent consumption."""
        return self.model_dump(mode="json")


def validate_case_payload(payload: dict[str, Any]) -> HealthcareCase:
    """Validate and return a typed healthcare case."""
    return HealthcareCase.model_validate(payload)
