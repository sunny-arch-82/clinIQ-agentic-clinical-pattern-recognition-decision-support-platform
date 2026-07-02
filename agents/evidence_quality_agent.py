"""Evidence Quality Agent.

This agent reviews online evidence before the Decision Agent runs.

It checks real retrieved evidence only.
Verification-gap rows are ignored while judging evidence quality.
"""

from __future__ import annotations

from typing import Any

from models.decision import EvidenceItem, RetrievedEvidence


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


def _is_gap_item(item: EvidenceItem) -> bool:
    return bool((item.metadata or {}).get("verification_gap"))


def _real_items(evidence: RetrievedEvidence) -> list[EvidenceItem]:
    return [
        item
        for item in evidence.all_items()
        if not _is_gap_item(item)
    ]


def _real_text(evidence: RetrievedEvidence) -> str:
    chunks: list[str] = []

    for item in _real_items(evidence):
        chunks.append(item.collection)
        chunks.append(item.source)
        chunks.append(item.retrieved_for)
        chunks.append(item.content)
        chunks.append(item.url or "")

    return " ".join(chunks).lower()


def _has_real_items(items: list[EvidenceItem]) -> bool:
    return any(not _is_gap_item(item) for item in items)


def _exact_match_found(text: str, values: list[str]) -> bool:
    for value in values:
        clean = _safe(value).lower()
        if clean and clean in text:
            return True

    return False


def _weak_sources_found(evidence: RetrievedEvidence) -> bool:
    for item in _real_items(evidence):
        text = f"{item.source} {item.content}".lower()

        if any(term in text for term in WEAK_SOURCE_TERMS):
            return True

    return False


def review_evidence(
    case: dict[str, Any],
    evidence: RetrievedEvidence,
    retrieval_attempts: int,
) -> dict[str, Any]:
    payer = case.get("payer") or {}
    provider = case.get("provider") or {}

    text = _real_text(evidence)

    has_clinical = _has_real_items(evidence.clinical)
    has_policy = _has_real_items(evidence.policies)
    has_coding = _has_real_items(evidence.coding)
    has_medical_necessity = _has_real_items(evidence.medical_necessity)

    payer_found = _exact_match_found(
        text,
        [
            payer.get("name"),
            payer.get("payer_id"),
            payer.get("policy_id"),
        ],
    )

    provider_found = _exact_match_found(
        text,
        [
            provider.get("name"),
            provider.get("npi"),
        ],
    )

    authorization_language_found = any(
        phrase in text
        for phrase in [
            "prior authorization",
            "preauthorization",
            "authorization required",
            "authorization is required",
            "emergency exception",
        ]
    )

    weak_sources_found = _weak_sources_found(evidence)

    gaps: list[str] = []

    if not has_clinical:
        gaps.append("Clinical evidence was not strong enough.")
    if not has_policy:
        gaps.append("Policy / coverage evidence was not strong enough.")
    if not has_coding:
        gaps.append("Coding evidence was not strong enough.")
    if not has_medical_necessity:
        gaps.append("Medical necessity evidence was not strong enough.")
    if not payer_found:
        gaps.append("Exact public payer or policy evidence was not found.")
    if not provider_found:
        gaps.append("Exact public provider / NPI evidence was not found.")
    if not authorization_language_found:
        gaps.append("Prior authorization status was not publicly verifiable.")
    if weak_sources_found:
        gaps.append("Weak or generic public sources were found and should be replaced if possible.")

    retry_needed = retrieval_attempts < 2 and (
        not has_policy
        or not has_medical_necessity
        or not payer_found
        or not provider_found
        or weak_sources_found
    )

    return {
        "has_clinical": has_clinical,
        "has_policy": has_policy,
        "has_coding": has_coding,
        "has_medical_necessity": has_medical_necessity,
        "payer_found": payer_found,
        "provider_found": provider_found,
        "authorization_language_found": authorization_language_found,
        "weak_sources_found": weak_sources_found,
        "gaps": gaps,
        "retry_needed": retry_needed,
    }


def build_retry_query_plan(case: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    payer = case.get("payer") or {}
    provider = case.get("provider") or {}
    claim = case.get("claim") or {}
    encounter = case.get("encounter") or {}

    diagnoses = case.get("diagnoses") or []
    procedures = case.get("procedures") or []
    documents = case.get("supporting_documents") or []

    diagnosis_text = " ".join(
        f"{item.get('code', '')} {item.get('description', '')}"
        for item in diagnoses
    )

    procedure_text = " ".join(
        f"{item.get('code', '')} {item.get('description', '')}"
        for item in procedures
    )

    document_text = " ".join(
        f"{item.get('document_type', '')} {item.get('title', '')} {item.get('summary', '')}"
        for item in documents
    )

    payer_name = _safe(payer.get("name"))
    payer_id = _safe(payer.get("payer_id"))
    plan_type = _safe(payer.get("plan_type"))
    policy_id = _safe(payer.get("policy_id"))

    provider_name = _safe(provider.get("name"))
    provider_npi = _safe(provider.get("npi"))
    provider_specialty = _safe(provider.get("specialty"))

    claim_type = _safe(claim.get("claim_type"))
    pos = _safe(encounter.get("place_of_service"))

    queries = {
        "clinical": [
            f"{diagnosis_text} {procedure_text} clinical guideline site:nih.gov OR site:ncbi.nlm.nih.gov OR site:cdc.gov",
            f"{diagnosis_text} {procedure_text} standard of care clinical documentation",
        ],
        "policies": [
            f'"{payer_name}" "{payer_id}" "{plan_type}" "{policy_id}" medical policy',
            f'"{policy_id}" payer coverage policy',
            f"{procedure_text} {diagnosis_text} prior authorization coverage policy",
            f"{claim_type} place of service {pos} inpatient coverage policy",
            f'"{provider_name}" "{provider_npi}" provider NPI hospital facility',
        ],
        "coding": [
            f"{diagnosis_text} {procedure_text} CPT ICD-10 billing code CMS AAPC",
        ],
        "medical_necessity": [
            f"{diagnosis_text} {procedure_text} {document_text} medical necessity documentation requirements",
            f"{procedure_text} {diagnosis_text} payer medical necessity criteria",
        ],
        "historical": [],
    }

    return {
        "queries": queries,
        "reason": "Evidence Quality Agent requested one retry because required evidence was weak or missing.",
        "previous_gaps": review.get("gaps", []),
    }


def _gap_exists(evidence: RetrievedEvidence, collection: str) -> bool:
    for item in evidence.all_items():
        if item.collection == collection and _is_gap_item(item):
            return True

    return False


def append_gap_items(
    evidence: RetrievedEvidence,
    case: dict[str, Any],
    review: dict[str, Any],
) -> RetrievedEvidence:
    payer = case.get("payer") or {}
    provider = case.get("provider") or {}

    if not review.get("payer_found") and not _gap_exists(evidence, "Payer Policy Public Check"):
        evidence.policies.append(
            EvidenceItem(
                collection="Payer Policy Public Check",
                source="No exact public payer / policy source found",
                retrieved_for="Payer verification",
                content=(
                    f"No exact public online source was found for payer "
                    f"'{_safe(payer.get('name'))}', plan '{_safe(payer.get('plan_type'))}', "
                    f"or policy ID '{_safe(payer.get('policy_id'))}'. "
                    "Reviewer should verify plan benefits, exclusions, coverage limits, and policy rules internally."
                ),
                url=None,
                internal_score=0.0,
                metadata={"verification_gap": True},
            )
        )

    if not review.get("provider_found") and not _gap_exists(evidence, "Provider Public Check"):
        evidence.policies.append(
            EvidenceItem(
                collection="Provider Public Check",
                source="No exact public provider / NPI source found",
                retrieved_for="Provider verification",
                content=(
                    f"No exact public online source was found for provider "
                    f"'{_safe(provider.get('name'))}' with NPI '{_safe(provider.get('npi'))}'. "
                    "Reviewer should verify facility status, network status, billing provider role, "
                    "rendering provider role, and specialty alignment internally."
                ),
                url=None,
                internal_score=0.0,
                metadata={"verification_gap": True},
            )
        )

    if not review.get("authorization_language_found") and not _gap_exists(evidence, "Authorization Status Check"):
        evidence.policies.append(
            EvidenceItem(
                collection="Authorization Status Check",
                source="No claim-specific authorization record available from public web search",
                retrieved_for="Authorization verification",
                content=(
                    "Prior authorization status is claim/member-specific and is normally not publicly available. "
                    "Reviewer should verify authorization, emergency exception, admission status, and plan rules internally."
                ),
                url=None,
                internal_score=0.0,
                metadata={"verification_gap": True},
            )
        )

    return evidence


def merge_evidence(primary: RetrievedEvidence, secondary: RetrievedEvidence) -> RetrievedEvidence:
    seen: set[tuple[str, str, str]] = set()

    for item in primary.all_items():
        seen.add((item.collection, item.source, item.retrieved_for))

    groups = [
        (primary.clinical, secondary.clinical),
        (primary.policies, secondary.policies),
        (primary.coding, secondary.coding),
        (primary.medical_necessity, secondary.medical_necessity),
        (primary.historical, secondary.historical),
    ]

    for target, incoming_items in groups:
        for item in incoming_items:
            key = (item.collection, item.source, item.retrieved_for)

            if key in seen:
                continue

            seen.add(key)
            target.append(item)

    return primary