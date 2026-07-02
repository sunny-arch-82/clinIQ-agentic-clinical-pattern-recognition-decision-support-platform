"""Pattern Recognition Agent.

Detects useful healthcare claim patterns from structured JSON.

This agent does not make claim decisions.
It only identifies review patterns, alignment signals, and documentation gaps.
"""

from __future__ import annotations

from typing import Any

from models.decision import PatternAnalysis


CARDIAC_TERMS = {
    "mi",
    "myocardial",
    "infarction",
    "stemi",
    "nstemi",
    "pci",
    "stent",
    "angioplasty",
    "coronary",
    "cardiac",
}

STROKE_TERMS = {
    "stroke",
    "cerebral",
    "infarction",
    "ct head",
    "critical care",
    "neuro",
    "neurology",
}

RESPIRATORY_TERMS = {
    "pneumonia",
    "asthma",
    "copd",
    "spirometry",
    "chest",
    "x-ray",
    "pulmonary",
    "respiratory",
}

ORTHOPEDIC_TERMS = {
    "fracture",
    "knee",
    "hip",
    "meniscus",
    "arthroscopy",
    "radius",
    "orthopedic",
    "x-ray",
    "mri",
}


def _safe(value: Any) -> str:
    return str(value or "").strip()


def _lower_case_text(case: dict[str, Any]) -> str:
    chunks: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif value is not None:
            chunks.append(str(value))

    walk(case)
    return " ".join(chunks).lower()


def _extract_diagnoses(case: dict[str, Any]) -> list[dict[str, Any]]:
    return case.get("diagnoses") or []


def _extract_procedures(case: dict[str, Any]) -> list[dict[str, Any]]:
    procedures = list(case.get("procedures") or [])

    claim = case.get("claim") or {}
    for line in claim.get("lines", []) or []:
        procedure_code = line.get("procedure_code")
        if procedure_code:
            procedures.append(
                {
                    "code": procedure_code,
                    "description": line.get("description", ""),
                }
            )

    return procedures


def _extract_documents(case: dict[str, Any]) -> list[dict[str, Any]]:
    return case.get("supporting_documents") or []


def _primary_diagnosis(case: dict[str, Any]) -> str:
    diagnoses = _extract_diagnoses(case)

    if not diagnoses:
        return "No diagnosis code found"

    primary = diagnoses[0]

    return f"{_safe(primary.get('code'))} - {_safe(primary.get('description'))}"


def _detect_named_patterns(case: dict[str, Any]) -> list[str]:
    text = _lower_case_text(case)
    patterns: list[str] = []

    if any(term in text for term in CARDIAC_TERMS):
        patterns.append(
            "Emergency inpatient cardiac intervention pattern detected: cardiac diagnosis/procedure terms appear aligned with acute cardiac care review."
        )

    if any(term in text for term in STROKE_TERMS):
        patterns.append(
            "Emergency neurologic care pattern detected: stroke/neuroimaging/critical-care terms require severity and intensity documentation review."
        )

    if any(term in text for term in RESPIRATORY_TERMS):
        patterns.append(
            "Respiratory diagnostic or treatment pattern detected: respiratory diagnosis/procedure terms require documentation and medical necessity review."
        )

    if any(term in text for term in ORTHOPEDIC_TERMS):
        patterns.append(
            "Orthopedic procedure pattern detected: imaging, operative notes, and medical necessity documentation should be reviewed."
        )

    return patterns


def _diagnosis_procedure_relationships(case: dict[str, Any]) -> list[str]:
    diagnoses = _extract_diagnoses(case)
    procedures = _extract_procedures(case)

    diagnosis_count = len([item for item in diagnoses if item.get("code")])
    procedure_count = len([item for item in procedures if item.get("code")])

    relationships = [
        f"{diagnosis_count} diagnosis code(s) linked to {procedure_count} procedure code(s)."
    ]

    if diagnoses and procedures:
        relationships.append("Diagnosis-to-procedure mapping is available and requires evidence review.")
    elif diagnoses and not procedures:
        relationships.append("Diagnosis information is present, but no procedure code was found.")
    elif procedures and not diagnoses:
        relationships.append("Procedure information is present, but no diagnosis code was found.")

    return relationships


def _document_patterns(case: dict[str, Any]) -> list[str]:
    documents = _extract_documents(case)
    points: list[str] = []

    points.append(f"{len(documents)} supporting document(s) available for review.")

    text = _lower_case_text(case)

    if "operative" in text or "procedure note" in text:
        points.append("Procedure documentation pattern detected: operative/procedure note is referenced.")

    if "mri" in text or "x-ray" in text or "ct" in text or "imaging" in text:
        points.append("Imaging documentation pattern detected: imaging evidence is referenced.")

    if "ekg" in text or "ecg" in text or "cath" in text:
        points.append("Cardiac supporting documentation pattern detected: EKG/ECG or cath-related record is referenced.")

    if "admission" in text or "h&p" in text or "history and physical" in text:
        points.append("Admission documentation pattern detected: admission or H&P record is referenced.")

    return points


def _alignment_checks(case: dict[str, Any]) -> list[str]:
    payer = case.get("payer") or {}
    provider = case.get("provider") or {}
    encounter = case.get("encounter") or {}
    claim = case.get("claim") or {}

    checks: list[str] = []

    checks.append(f"Primary diagnosis: {_primary_diagnosis(case)}")

    if claim.get("claim_type") and encounter.get("place_of_service"):
        checks.append(
            f"Claim type / place of service alignment requires review: {claim.get('claim_type')} / POS {encounter.get('place_of_service')}."
        )

    if payer.get("policy_id"):
        checks.append(
            f"Payer policy reference present in JSON: {payer.get('policy_id')}."
        )

    if provider.get("npi"):
        checks.append(
            f"Provider/NPI reference present in JSON: {provider.get('name')} / {provider.get('npi')}."
        )

    return checks


def _missing_information(case: dict[str, Any]) -> list[str]:
    missing: list[str] = []

    payer = case.get("payer") or {}
    provider = case.get("provider") or {}
    documents = _extract_documents(case)

    text = _lower_case_text(case)

    if not payer.get("policy_id"):
        missing.append("Payer policy ID is missing from the claim JSON.")

    if not provider.get("npi"):
        missing.append("Provider NPI is missing from the claim JSON.")

    if not documents:
        missing.append("Supporting clinical documents are missing.")

    if "authorization" not in text and "prior auth" not in text:
        missing.append("Prior authorization status is not available in the claim JSON.")

    if "benefit" not in text and "eligibility" not in text:
        missing.append("Member benefit / eligibility verification is not available in the claim JSON.")

    return missing


def analyze_patterns(case: dict[str, Any]) -> PatternAnalysis:
    relationships = _diagnosis_procedure_relationships(case)
    clinical_patterns = _detect_named_patterns(case) + _document_patterns(case)
    alignment_checks = _alignment_checks(case)
    missing_information = _missing_information(case)

    summary = (
        f"Case {(case.get('case_information') or {}).get('case_id', 'unknown')} analyzed. "
        f"Detected {len(clinical_patterns)} pattern signal(s), "
        f"{len(relationships)} relationship signal(s), and "
        f"{len(missing_information)} documentation or verification gap(s)."
    )

    return PatternAnalysis(
        summary=summary,
        relationships=relationships,
        clinical_patterns=clinical_patterns,
        alignment_checks=alignment_checks,
        missing_information=missing_information,
        inconsistencies=[],
    )