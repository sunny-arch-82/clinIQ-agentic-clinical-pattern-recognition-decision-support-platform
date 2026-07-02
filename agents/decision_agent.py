"""Decision Intelligence Agent.

Layered output:
1. Clinical Evidence Position
2. Verification Status
3. Final Reviewer Recommendation
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from models.decision import (
    DecisionRecommendation,
    PatternAnalysis,
    RecommendationType,
    RetrievedEvidence,
    VerificationResult,
)
from utils.config_loader import get_project_root
from utils.llm_client import call_llm, extract_json, rough_token_estimate


PROMPT_PATH = get_project_root() / "prompts" / "decision.txt"

TARGET_TOTAL_TOKENS = 10800
LLM_OUTPUT_TOKENS = 1300

MAX_REAL_EVIDENCE_ITEMS = 10
MAX_GAP_ITEMS = 3
MAX_EVIDENCE_EXCERPT_CHARS = 1400

MIN_REAL_EVIDENCE_ITEMS = 6
MIN_EVIDENCE_EXCERPT_CHARS = 700

MAX_DECISION_SUPPORT_ITEMS = 5
MAX_MISSING_ITEMS = 5
MAX_ACTION_ITEMS = 4

DROP_COLLECTIONS = {
    "Public Reference Evidence",
    "Public Reference / Similar Case Evidence",
}

WEAK_SOURCE_TERMS = [
    "brainly",
    "quizlet",
    "coursehero",
    "chegg",
    "reddit",
    "quora",
    "emr software",
    "ehr software",
    "software",
    "chronic rhinitis",
    "pediatric",
    "ultimate guide",
    "billing services",
    "medical billing services",
]


def _safe(value: Any) -> str:
    return str(value or "").strip()


def _trim(value: Any, max_chars: int) -> str:
    text = _safe(value).replace("\n", " ")
    text = " ".join(text.split())

    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"

    return text


def _is_gap_item(item: Any) -> bool:
    return bool((item.metadata or {}).get("verification_gap"))


def _is_weak_source(item: Any) -> bool:
    text = f"{item.source} {item.content}".lower()
    return any(term in text for term in WEAK_SOURCE_TERMS)


def _claim_review_points(case: dict[str, Any]) -> dict[str, Any]:
    case_info = case.get("case_information") or {}
    payer = case.get("payer") or {}
    provider = case.get("provider") or {}
    encounter = case.get("encounter") or {}
    claim = case.get("claim") or {}

    return {
        "case_information": {
            "case_id": case_info.get("case_id"),
            "created_at": case_info.get("created_at"),
        },
        "payer": {
            "name": payer.get("name"),
            "payer_id": payer.get("payer_id"),
            "plan_type": payer.get("plan_type"),
            "policy_id": payer.get("policy_id"),
        },
        "provider": {
            "name": provider.get("name"),
            "npi": provider.get("npi"),
            "specialty": provider.get("specialty"),
        },
        "encounter": {
            "encounter_type": encounter.get("encounter_type"),
            "place_of_service": encounter.get("place_of_service"),
            "admission_date": encounter.get("admission_date"),
            "discharge_date": encounter.get("discharge_date"),
        },
        "claim": {
            "claim_id": claim.get("claim_id"),
            "claim_type": claim.get("claim_type"),
            "total_billed": claim.get("total_billed"),
            "service_from": claim.get("service_from"),
            "service_to": claim.get("service_to"),
            "lines": claim.get("lines", []),
        },
        "diagnoses": case.get("diagnoses", []),
        "procedures": case.get("procedures", []),
        "supporting_documents": case.get("supporting_documents", []),
    }


def _patterns_for_llm(pattern: PatternAnalysis) -> dict[str, Any]:
    return {
        "summary": _trim(pattern.summary, 700),
        "relationships": [_trim(item, 260) for item in pattern.relationships[:6]],
        "clinical_patterns": [_trim(item, 260) for item in pattern.clinical_patterns[:6]],
        "alignment_checks": [_trim(item, 260) for item in pattern.alignment_checks[:6]],
        "missing_information": [_trim(item, 260) for item in pattern.missing_information[:6]],
        "inconsistencies": [
            {
                "category": finding.category,
                "severity": finding.severity,
                "message": _trim(finding.message, 260),
                "related_fields": finding.related_fields,
            }
            for finding in pattern.inconsistencies[:6]
        ],
    }


def _compact_real_evidence_with_limits(
    evidence: RetrievedEvidence,
    max_items: int,
    excerpt_chars: int,
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    sorted_items = sorted(
        evidence.all_items(),
        key=lambda item: getattr(item, "internal_score", 0.0),
        reverse=True,
    )

    seen: set[tuple[str, str]] = set()

    for item in sorted_items:
        if _is_gap_item(item):
            continue

        if item.collection in DROP_COLLECTIONS:
            continue

        if _is_weak_source(item):
            continue

        key = (item.collection, item.source)
        if key in seen:
            continue

        seen.add(key)

        items.append(
            {
                "category": item.collection,
                "source": _trim(item.source, 180),
                "url": item.url or "",
                "retrieved_for": _trim(item.retrieved_for, 120),
                "evidence_excerpt": _trim(item.content, excerpt_chars),
            }
        )

        if len(items) >= max_items:
            break

    return items


def _compact_gap_evidence(evidence: RetrievedEvidence) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in evidence.all_items():
        if not _is_gap_item(item):
            continue

        if item.collection in seen:
            continue

        seen.add(item.collection)

        gaps.append(
            {
                "gap": item.collection,
                "source": item.source,
                "reviewer_action": _trim(item.content, 420),
            }
        )

        if len(gaps) >= MAX_GAP_ITEMS:
            break

    return gaps


def _verification_for_llm(verification_result: VerificationResult | None) -> dict[str, Any]:
    if verification_result is None:
        return {
            "overall_status": "Internal Verification Required",
            "summary": "Verification Agent did not return a result.",
            "checks": [],
        }

    return {
        "overall_status": verification_result.overall_status,
        "summary": _trim(verification_result.summary, 600),
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "finding": _trim(check.finding, 300),
                "reviewer_action": _trim(check.reviewer_action, 300),
            }
            for check in verification_result.checks[:5]
        ],
    }


def _build_llm_payload(
    case: dict[str, Any],
    pattern: PatternAnalysis,
    evidence: RetrievedEvidence,
    verification_result: VerificationResult | None,
    evidence_items: int,
    excerpt_chars: int,
) -> dict[str, Any]:
    return {
        "claim_review_points": _claim_review_points(case),
        "patterns_recognized": _patterns_for_llm(pattern),
        "online_evidence": {
            "retrieved_sources": _compact_real_evidence_with_limits(
                evidence=evidence,
                max_items=evidence_items,
                excerpt_chars=excerpt_chars,
            ),
            "public_verification_gaps": _compact_gap_evidence(evidence),
        },
        "verification_result": _verification_for_llm(verification_result),
        "decision_rules": [
            "This is decision-support only. Do not approve, deny, or adjudicate the claim.",
            "Separate clinical evidence position from verification status.",
            "Clinical Evidence Position should reflect whether clinical/coding/medical-necessity evidence supports the claim.",
            "Verification Status should reflect payer/provider/authorization checks.",
            "If clinical evidence is aligned but payer/provider/authorization checks require internal verification, final recommendation should usually be Requires Manual Review.",
            "Do not mention confidence levels, percentages, internal scores, or ranking scores.",
            "Keep the language concise, professional, and reviewer-facing.",
        ],
        "allowed_recommendations": [item.value for item in RecommendationType],
        "required_json_schema": {
            "clinical_evidence_position": "Likely Supported / Additional Documentation Needed / Evidence Insufficient / Likely Not Supported",
            "verification_status": "Public Verification Completed / Internal Verification Required / Not Publicly Verifiable",
            "recommendation": "one allowed recommendation string",
            "final_recommendation_basis": "one concise explanation of why the final recommendation was selected",
            "reasoning_summary": "3 to 5 concise sentences",
            "decision_support": ["max 5 key evidence points"],
            "missing_information": ["max 5 evidence gaps or missing facts"],
            "suggested_next_actions": ["max 4 reviewer actions"],
            "reviewer_notes": "one short sentence",
        },
    }


def _fit_payload_under_token_budget(
    case: dict[str, Any],
    pattern: PatternAnalysis,
    evidence: RetrievedEvidence,
    verification_result: VerificationResult | None,
    system_prompt: str,
) -> tuple[dict[str, Any], int, int, int]:
    evidence_items = MAX_REAL_EVIDENCE_ITEMS
    excerpt_chars = MAX_EVIDENCE_EXCERPT_CHARS

    while True:
        payload = _build_llm_payload(
            case=case,
            pattern=pattern,
            evidence=evidence,
            verification_result=verification_result,
            evidence_items=evidence_items,
            excerpt_chars=excerpt_chars,
        )

        prompt_text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

        estimated_total_tokens = (
            rough_token_estimate(system_prompt)
            + rough_token_estimate(prompt_text)
            + LLM_OUTPUT_TOKENS
        )

        if estimated_total_tokens <= TARGET_TOTAL_TOKENS:
            return payload, evidence_items, excerpt_chars, estimated_total_tokens

        if excerpt_chars > MIN_EVIDENCE_EXCERPT_CHARS:
            excerpt_chars -= 150
            continue

        if evidence_items > MIN_REAL_EVIDENCE_ITEMS:
            evidence_items -= 1
            excerpt_chars = MAX_EVIDENCE_EXCERPT_CHARS
            continue

        return payload, evidence_items, excerpt_chars, estimated_total_tokens


def _verification_has_internal_requirements(verification_result: VerificationResult | None) -> bool:
    if verification_result is None:
        return True

    return verification_result.overall_status != "Public Verification Completed"


def _fallback_decision(
    pattern: PatternAnalysis,
    evidence: RetrievedEvidence,
    verification_result: VerificationResult | None,
) -> DecisionRecommendation:
    missing: list[str] = list(pattern.missing_information)

    real_items = [
        item
        for item in evidence.all_items()
        if not _is_gap_item(item)
        and item.collection not in DROP_COLLECTIONS
        and not _is_weak_source(item)
    ]

    gap_items = [
        item
        for item in evidence.all_items()
        if _is_gap_item(item)
    ]

    for item in gap_items:
        missing.append(_trim(item.content, 240))

    if verification_result:
        for check in verification_result.checks:
            if check.status != "Verified from public evidence":
                missing.append(f"{check.name}: {check.reviewer_action}")

    has_internal_verification = _verification_has_internal_requirements(verification_result)

    if not real_items:
        clinical_position = "Evidence Insufficient"
        recommendation = RecommendationType.EVIDENCE_INSUFFICIENT
    elif has_internal_verification:
        clinical_position = "Likely Supported"
        recommendation = RecommendationType.REQUIRES_MANUAL_REVIEW
    elif missing:
        clinical_position = "Additional Documentation Needed"
        recommendation = RecommendationType.ADDITIONAL_DOCUMENTATION
    else:
        clinical_position = "Likely Supported"
        recommendation = RecommendationType.LIKELY_SUPPORTED

    decision_support = [
        f"{item.collection}: {item.source} was used as online evidence."
        for item in real_items[:MAX_DECISION_SUPPORT_ITEMS]
    ]

    if not decision_support:
        decision_support = [
            "Pattern analysis completed, but strong online evidence was limited.",
        ]

    verification_status = (
        verification_result.overall_status
        if verification_result is not None
        else "Internal Verification Required"
    )

    return DecisionRecommendation(
        recommendation=recommendation,
        clinical_evidence_position=clinical_position,
        verification_status=verification_status,
        final_recommendation_basis=(
            "The clinical/coding evidence may support the claim, but payer, provider, or authorization "
            "verification requires internal review."
        ),
        reasoning_summary=(
            "The system completed claim pattern analysis, online evidence retrieval, and verification review. "
            "The configured LLM was unavailable or rate-limited, so a conservative reviewer-facing recommendation was generated."
        ),
        decision_support=decision_support[:MAX_DECISION_SUPPORT_ITEMS],
        missing_information=[_trim(item, 240) for item in missing[:MAX_MISSING_ITEMS]],
        suggested_next_actions=[
            "Verify payer-specific policy, plan benefits, and authorization status internally.",
            "Confirm provider/NPI and network status internally.",
            "Review supporting clinical documentation against the billed diagnosis and procedure.",
        ],
        reviewer_notes="Fallback output only. Human reviewer remains the final authority.",
    )


def _normalize_decision_payload(parsed: dict[str, Any]) -> DecisionRecommendation:
    recommendation_value = parsed.get(
        "recommendation",
        RecommendationType.REQUIRES_MANUAL_REVIEW.value,
    )

    try:
        recommendation = RecommendationType(recommendation_value)
    except Exception:
        recommendation = RecommendationType.REQUIRES_MANUAL_REVIEW

    return DecisionRecommendation(
        recommendation=recommendation,
        clinical_evidence_position=_trim(
            parsed.get("clinical_evidence_position", "Not Assessed"),
            180,
        ),
        verification_status=_trim(
            parsed.get("verification_status", "Internal Verification Required"),
            180,
        ),
        final_recommendation_basis=_trim(
            parsed.get("final_recommendation_basis", ""),
            500,
        ),
        reasoning_summary=_trim(parsed.get("reasoning_summary", ""), 1200),
        decision_support=[
            _trim(item, 320)
            for item in parsed.get("decision_support", [])[:MAX_DECISION_SUPPORT_ITEMS]
        ],
        missing_information=[
            _trim(item, 280)
            for item in parsed.get("missing_information", [])[:MAX_MISSING_ITEMS]
        ],
        suggested_next_actions=[
            _trim(item, 280)
            for item in parsed.get("suggested_next_actions", [])[:MAX_ACTION_ITEMS]
        ],
        reviewer_notes=_trim(parsed.get("reviewer_notes", ""), 320),
    )


def generate_decision(
    case: dict[str, Any],
    pattern: PatternAnalysis,
    evidence: RetrievedEvidence,
    verification_result: VerificationResult | None = None,
    override_model: str | None = None,
) -> DecisionRecommendation:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    try:
        user_payload, evidence_items, excerpt_chars, estimated_total_tokens = _fit_payload_under_token_budget(
            case=case,
            pattern=pattern,
            evidence=evidence,
            verification_result=verification_result,
            system_prompt=system_prompt,
        )

        compact_prompt = json.dumps(user_payload, separators=(",", ":"), ensure_ascii=False)

        logger.info(
            "Decision Agent prompt budget: "
            f"estimated_total_tokens≈{estimated_total_tokens}, "
            f"evidence_items={evidence_items}, "
            f"excerpt_chars={excerpt_chars}, "
            f"target_total_tokens={TARGET_TOTAL_TOKENS}"
        )

        raw = call_llm(
            system_prompt=system_prompt,
            user_prompt=compact_prompt,
            override_model=override_model,
            max_tokens=LLM_OUTPUT_TOKENS,
            temperature=0.2,
        )

        parsed = extract_json(raw)
        return _normalize_decision_payload(parsed)

    except Exception as exc:
        logger.warning(f"Decision Agent fallback used because LLM call failed: {exc}")
        return _fallback_decision(pattern, evidence, verification_result)