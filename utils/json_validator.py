"""JSON validation and normalization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.case_schema import HealthcareCase, validate_case_payload


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_and_normalize(payload: dict[str, Any]) -> HealthcareCase:
    case = validate_case_payload(payload)
    return case


def validate_json_file(path: Path) -> HealthcareCase:
    payload = load_json_file(path)
    return validate_and_normalize(payload)
