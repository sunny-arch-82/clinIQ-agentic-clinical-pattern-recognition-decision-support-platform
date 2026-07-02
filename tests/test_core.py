"""Tests for ClinIQ."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.pattern_agent import analyze_patterns
from demo_cases.generate_demo_cases import write_demo_cases
from utils.json_validator import validate_and_normalize


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_demo_case_generation():
    output_dir = PROJECT_ROOT / "demo_cases"
    write_demo_cases(output_dir)
    files = list(output_dir.glob("*.json"))
    assert len(files) >= 10


def test_json_validation():
    write_demo_cases(PROJECT_ROOT / "demo_cases")
    sample = PROJECT_ROOT / "demo_cases" / "diabetes.json"
    payload = json.loads(sample.read_text(encoding="utf-8"))
    case = validate_and_normalize(payload)
    assert case.case_information.case_id == "CASE-DM-001"


def test_pattern_agent():
    write_demo_cases(PROJECT_ROOT / "demo_cases")
    payload = json.loads((PROJECT_ROOT / "demo_cases" / "pneumonia.json").read_text(encoding="utf-8"))
    pattern = analyze_patterns(payload)
    assert pattern.summary
    assert pattern.relationships
