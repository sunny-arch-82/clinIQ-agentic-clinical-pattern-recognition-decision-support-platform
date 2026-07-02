"""Query Planner Agent.

This agent makes the workflow more agentic by planning what the online
retriever should search for before retrieval happens.

It is deterministic and does not require an LLM.
"""

from __future__ import annotations

from typing import Any


def _safe(value: Any) -> str:
    return str(value or "").strip()


def _join(values: list[str]) -> str:
    return " ".join(value for value in values if value)


def build_query_plan(case: dict[str, Any], pattern_analysis: Any | None = None) -> dict[str, Any]:
    case_info = case.get("case_information") or {}
    payer = case.get("payer") or {}
    provider = case.get("provider") or {}
    encounter = case.get("encounter") or {}
    claim = case.get("claim") or {}

    diagnoses = case.get("diagnoses") or []
    procedures = case.get("procedures") or []
    documents = case.get("supporting_documents") or []
    claim_lines = claim.get("lines") or []

    diagnosis_codes = [_safe(item.get("code")) for item in diagnoses]
    diagnosis_descriptions = [_safe(item.get("description")) for item in diagnoses]

    procedure_codes = [_safe(item.get("code")) for item in procedures]
    procedure_descriptions = [_safe(item.get("description")) for item in procedures]

    line_codes: list[str] = []
    diagnosis_pointers: list[str] = []

    for line in claim_lines:
        if line.get("procedure_code"):
            line_codes.append(_safe(line.get("procedure_code")))

        for pointer in line.get("diagnosis_pointers", []) or []:
            diagnosis_pointers.append(_safe(pointer))

    for code in line_codes:
        if code and code not in procedure_codes:
            procedure_codes.append(code)

    document_text = _join(
        [
            *[_safe(item.get("document_type")) for item in documents],
            *[_safe(item.get("title")) for item in documents],
            *[_safe(item.get("summary")) for item in documents],
        ]
    )

    diagnosis_text = _join(diagnosis_codes + diagnosis_descriptions)
    procedure_text = _join(procedure_codes + procedure_descriptions)

    payer_context = _join(
        [
            _safe(payer.get("name")),
            _safe(payer.get("payer_id")),
            _safe(payer.get("plan_type")),
            _safe(payer.get("policy_id")),
        ]
    )

    provider_context = _join(
        [
            _safe(provider.get("name")),
            _safe(provider.get("npi")),
            _safe(provider.get("specialty")),
        ]
    )

    encounter_context = _join(
        [
            _safe(claim.get("claim_type")),
            _safe(encounter.get("encounter_type")),
            f"place of service {_safe(encounter.get('place_of_service'))}",
            f"admission {_safe(encounter.get('admission_date'))}",
            f"discharge {_safe(encounter.get('discharge_date'))}",
            f"service dates {_safe(claim.get('service_from'))} {_safe(claim.get('service_to'))}",
            f"total billed {_safe(claim.get('total_billed'))}",
        ]
    )

    line_context = _join(
        [
            f"claim line procedure codes {_join(line_codes)}",
            f"diagnosis pointers {_join(diagnosis_pointers)}",
        ]
    )

    pattern_text = ""
    if pattern_analysis:
        try:
            pattern_text = " ".join(pattern_analysis.reviewer_points())
        except Exception:
            pattern_text = _safe(getattr(pattern_analysis, "summary", ""))

    queries = {
        "clinical": [
            f"{diagnosis_text} {procedure_text} clinical guideline treatment pathway",
            f"{diagnosis_text} {procedure_text} medical documentation standard of care",
        ],
        "policies": [
            f"{payer_context} {procedure_text} coverage policy prior authorization medical necessity",
            f"{payer_context} {encounter_context} provider manual medical policy",
            f"{procedure_text} {diagnosis_text} insurance coverage policy authorization",
        ],
        "coding": [
            f"{diagnosis_text} {procedure_text} ICD-10 CPT HCPCS coding reference",
            f"{line_context} diagnosis pointer coding claim line billing rules",
        ],
        "medical_necessity": [
            f"{diagnosis_text} {procedure_text} {document_text} medical necessity documentation requirements",
            f"{procedure_text} supporting records documentation clinical necessity {pattern_text}",
        ],
        "historical": [
            f"{diagnosis_text} {procedure_text} public case report medical necessity documentation",
            f"{diagnosis_text} {procedure_text} example claim review coding documentation",
        ],
    }

    required_checks = [
        "Clinical diagnosis/procedure support",
        "Coding support",
        "Medical necessity documentation",
        "Payer policy / coverage evidence",
        "Prior authorization or emergency exception evidence",
        "Provider / facility / NPI verification",
        "Claim line and diagnosis pointer support",
    ]

    return {
        "case_id": _safe(case_info.get("case_id")),
        "queries": queries,
        "required_checks": required_checks,
        "claim_focus": {
            "payer": payer_context,
            "provider": provider_context,
            "diagnosis": diagnosis_text,
            "procedure": procedure_text,
            "encounter": encounter_context,
            "claim_lines": line_context,
        },
    }
