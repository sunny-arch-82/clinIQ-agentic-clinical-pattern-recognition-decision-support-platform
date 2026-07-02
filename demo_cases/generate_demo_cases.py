"""Demo case factory for standardized healthcare JSON examples."""

from __future__ import annotations

import json
from pathlib import Path


def _base_case(
    case_id: str,
    disease: str,
    primary_dx: tuple[str, str],
    procedures: list[dict],
    docs: list[str],
    claim_total: float,
    payer: str = "National Health Plan",
) -> dict:
    return {
        "case_information": {
            "case_id": case_id,
            "schema_version": "1.0",
            "created_at": "2026-06-01",
            "case_type": "claim_review",
        },
        "patient": {
            "patient_id": f"PAT-{case_id}",
            "first_name": "Alex",
            "last_name": "Rivera",
            "date_of_birth": "1978-04-12",
            "gender": "F",
            "member_id": f"MBR-{case_id}",
        },
        "provider": {
            "provider_id": "PRV-1001",
            "name": "Metro General Hospital",
            "npi": "1234567890",
            "specialty": "Internal Medicine",
        },
        "payer": {
            "payer_id": "PAY-001",
            "name": payer,
            "plan_type": "PPO",
            "policy_id": "POL-STANDARD-2026",
        },
        "encounter": {
            "encounter_id": f"ENC-{case_id}",
            "encounter_type": "inpatient",
            "admission_date": "2026-05-15",
            "discharge_date": "2026-05-18",
            "place_of_service": "21",
        },
        "diagnoses": [
            {
                "code": primary_dx[0],
                "system": "ICD-10-CM",
                "description": primary_dx[1],
                "rank": 1,
                "is_primary": True,
            }
        ],
        "procedures": procedures,
        "medications": [],
        "labs": [],
        "claim": {
            "claim_id": f"CLM-{case_id}",
            "claim_type": "837I",
            "total_billed": claim_total,
            "service_from": "2026-05-15",
            "service_to": "2026-05-18",
            "lines": [
                {
                    "line_number": idx + 1,
                    "procedure_code": proc["code"],
                    "diagnosis_pointers": [1],
                    "billed_amount": proc.get("charge_amount", claim_total / max(len(procedures), 1)),
                }
                for idx, proc in enumerate(procedures)
            ],
        },
        "supporting_documents": [
            {
                "document_id": f"DOC-{case_id}-{idx + 1}",
                "document_type": "clinical_note",
                "title": title,
                "summary": summary,
                "date": "2026-05-16",
            }
            for idx, (title, summary) in enumerate(
                [(d, f"Clinical documentation supporting {disease} management.") for d in docs]
            )
        ],
    }


CASES = {
    "diabetes": _base_case(
        "CASE-DM-001",
        "Type 2 Diabetes Mellitus",
        ("E11.9", "Type 2 diabetes mellitus without complications"),
        [{"code": "99223", "system": "CPT", "description": "Initial hospital care", "charge_amount": 4200.0}],
        ["Endocrinology consult", "HbA1c trend note"],
        4200.0,
    ),
    "asthma": _base_case(
        "CASE-AST-002",
        "Asthma",
        ("J45.909", "Unspecified asthma, uncomplicated"),
        [{"code": "94010", "system": "CPT", "description": "Spirometry", "charge_amount": 850.0}],
        ["Pulmonary function report", "ED visit note"],
        850.0,
    ),
    "pneumonia": _base_case(
        "CASE-PNA-003",
        "Pneumonia",
        ("J18.9", "Pneumonia, unspecified organism"),
        [
            {"code": "71046", "system": "CPT", "description": "Chest X-ray", "charge_amount": 650.0},
            {"code": "99223", "system": "CPT", "description": "Initial hospital care", "charge_amount": 5100.0},
        ],
        ["Chest imaging report", "Admission H&P"],
        5750.0,
    ),
    "copd": _base_case(
        "CASE-COPD-004",
        "COPD",
        ("J44.1", "COPD with acute exacerbation"),
        [{"code": "94640", "system": "CPT", "description": "Nebulizer treatment", "charge_amount": 1200.0}],
        ["Pulmonology note", "ABG results"],
        1200.0,
    ),
    "stroke": _base_case(
        "CASE-STR-005",
        "Acute Ischemic Stroke",
        ("I63.9", "Cerebral infarction, unspecified"),
        [
            {"code": "70450", "system": "CPT", "description": "CT head without contrast", "charge_amount": 2200.0},
            {"code": "99291", "system": "CPT", "description": "Critical care first hour", "charge_amount": 6800.0},
        ],
        ["Neurology consult", "CT head report"],
        9000.0,
    ),
    "hip_replacement": _base_case(
        "CASE-HIP-006",
        "Hip Replacement",
        ("M16.11", "Primary osteoarthritis, right hip"),
        [{"code": "27130", "system": "CPT", "description": "Total hip arthroplasty", "charge_amount": 28500.0}],
        ["Orthopedic operative report", "PT evaluation"],
        28500.0,
    ),
    "heart_attack": _base_case(
        "CASE-MI-007",
        "Acute Myocardial Infarction",
        ("I21.3", "ST elevation MI of unspecified site"),
        [
            {"code": "92928", "system": "CPT", "description": "PCI with stent", "charge_amount": 18500.0},
            {"code": "99223", "system": "CPT", "description": "Initial hospital care", "charge_amount": 6200.0},
        ],
        ["Cardiology cath report", "EKG summary"],
        24700.0,
    ),
    "fracture": _base_case(
        "CASE-FX-008",
        "Distal Radius Fracture",
        ("S52.501A", "Unspecified fracture of right radius, initial encounter"),
        [{"code": "25607", "system": "CPT", "description": "Open treatment radius fracture", "charge_amount": 9800.0}],
        ["Orthopedic consult", "X-ray report"],
        9800.0,
    ),
    "sepsis": _base_case(
        "CASE-SEP-009",
        "Sepsis",
        ("A41.9", "Sepsis, unspecified organism"),
        [
            {"code": "99291", "system": "CPT", "description": "Critical care first hour", "charge_amount": 7200.0},
            {"code": "87040", "system": "CPT", "description": "Blood culture", "charge_amount": 450.0},
        ],
        ["ICU progress note", "Culture results"],
        7650.0,
    ),
    "knee_surgery": _base_case(
        "CASE-KNEE-010",
        "Knee Arthroscopy",
        ("M23.211", "Derangement of posterior horn of medial meniscus, right knee"),
        [{"code": "29881", "system": "CPT", "description": "Knee arthroscopy with meniscectomy", "charge_amount": 11200.0}],
        ["MRI knee report", "Operative note"],
        11200.0,
    ),
}


def write_demo_cases(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in CASES.items():
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    write_demo_cases(Path(__file__).resolve().parent / "demo_cases")
