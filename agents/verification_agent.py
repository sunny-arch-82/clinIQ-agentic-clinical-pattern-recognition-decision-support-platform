"""Verification Agent.

Separates public online evidence from private/internal verification.

Public web RAG can support:
- clinical logic
- ICD/CPT coding support
- general policy references
- medical necessity documentation expectations

Public web RAG usually cannot verify:
- exact member benefits
- exact payer policy ID
- provider network status
- billing/rendering provider contract status
- prior authorization records
"""

from __future__ import annotations

from typing import Any

from models.decision import RetrievedEvidence, VerificationCheck, VerificationResult


def _safe(value: Any) -> str:
    return str(value or "").strip()


def _is_gap_item(item: Any) -> bool:
    return bool((item.metadata or {}).get("verification_gap"))


def _real_evidence_text(evidence: RetrievedEvidence) -> str:
    chunks: list[str] = []

    for item in evidence.all_items():
        if _is_gap_item(item):
            continue

        chunks.extend(
            [
                item.collection,
                item.source,
                item.content,
                item.url or "",
            ]
        )

    return " ".join(chunks).lower()


def _contains_any(text: str, values: list[Any]) -> bool:
    for value in values:
        clean = _safe(value).lower()
        if clean and clean in text:
            return True

    return False


def _has_authorization_language(text: str) -> bool:
    phrases = [
        "prior authorization",
        "preauthorization",
        "authorization required",
        "authorization is required",
        "emergency exception",
        "no authorization required",
        "notification required",
    ]

    return any(phrase in text for phrase in phrases)


def _has_policy_language(text: str) -> bool:
    phrases = [
        "coverage",
        "medical policy",
        "clinical policy",
        "provider manual",
        "medical necessity",
        "covered",
        "not covered",
        "benefit",
        "benefits",
        "authorization",
        "reimbursement policy",
    ]

    return any(phrase in text for phrase in phrases)


def _has_provider_language(text: str) -> bool:
    phrases = [
        "npi",
        "national provider identifier",
        "provider",
        "facility",
        "hospital",
        "network",
    ]

    return any(phrase in text for phrase in phrases)


def run_verification(
    case: dict[str, Any],
    evidence: RetrievedEvidence,
    evidence_review: dict[str, Any] | None = None,
) -> VerificationResult:
    payer = case.get("payer") or {}
    provider = case.get("provider") or {}

    payer_name = _safe(payer.get("name"))
    payer_id = _safe(payer.get("payer_id"))
    plan_type = _safe(payer.get("plan_type"))
    policy_id = _safe(payer.get("policy_id"))

    provider_name = _safe(provider.get("name"))
    provider_npi = _safe(provider.get("npi"))

    text = _real_evidence_text(evidence)

    payer_exact_found = _contains_any(text, [payer_name, payer_id, policy_id])
    provider_exact_found = _contains_any(text, [provider_name, provider_npi])

    policy_language_found = _has_policy_language(text)
    provider_language_found = _has_provider_language(text)
    authorization_language_found = _has_authorization_language(text)

    checks: list[VerificationCheck] = []

    if payer_exact_found:
        checks.append(
            VerificationCheck(
                name="Payer Policy Verification",
                status="Verified from public evidence",
                finding=(
                    f"Public evidence contained an exact payer or policy reference for "
                    f"{payer_name or payer_id or policy_id}."
                ),
                reviewer_action="Confirm final plan benefits and member-specific limits internally.",
            )
        )
    elif policy_language_found:
        checks.append(
            VerificationCheck(
                name="Payer Policy Verification",
                status="Internal verification required",
                finding=(
                    f"General policy or coverage evidence was found online, but no exact public match "
                    f"was found for payer '{payer_name}', plan '{plan_type}', or policy ID '{policy_id}'."
                ),
                reviewer_action=(
                    "Verify payer-specific policy, member benefits, exclusions, and coverage limits "
                    "inside the payer portal or internal policy repository."
                ),
            )
        )
    else:
        checks.append(
            VerificationCheck(
                name="Payer Policy Verification",
                status="Not publicly verifiable",
                finding=(
                    f"No usable public payer-policy evidence was found for payer '{payer_name}', "
                    f"plan '{plan_type}', or policy ID '{policy_id}'."
                ),
                reviewer_action=(
                    "Use internal payer policy systems or direct payer verification before final claim action."
                ),
            )
        )

    if provider_exact_found:
        checks.append(
            VerificationCheck(
                name="Provider / Hospital Verification",
                status="Verified from public evidence",
                finding=(
                    f"Public evidence contained an exact provider or NPI reference for "
                    f"{provider_name or provider_npi}."
                ),
                reviewer_action="Confirm network and billing/rendering provider roles internally.",
            )
        )
    elif provider_language_found:
        checks.append(
            VerificationCheck(
                name="Provider / Hospital Verification",
                status="Internal verification required",
                finding=(
                    f"Provider-related public evidence was found, but no exact public match was found "
                    f"for provider '{provider_name}' with NPI '{provider_npi}'."
                ),
                reviewer_action=(
                    "Verify facility status, network status, NPI, billing provider role, rendering provider role, "
                    "and specialty alignment internally."
                ),
            )
        )
    else:
        checks.append(
            VerificationCheck(
                name="Provider / Hospital Verification",
                status="Not publicly verifiable",
                finding=(
                    f"No exact public provider/NPI evidence was found for '{provider_name}' "
                    f"with NPI '{provider_npi}'."
                ),
                reviewer_action=(
                    "Use internal provider-network and NPI validation systems before final claim action."
                ),
            )
        )

    if authorization_language_found:
        checks.append(
            VerificationCheck(
                name="Authorization Verification",
                status="Internal verification required",
                finding=(
                    "Public evidence contained authorization-related language, but claim-specific authorization "
                    "status is not publicly available."
                ),
                reviewer_action=(
                    "Verify prior authorization, emergency exception, admission notification, and plan-specific "
                    "authorization rules internally."
                ),
            )
        )
    else:
        checks.append(
            VerificationCheck(
                name="Authorization Verification",
                status="Not publicly verifiable",
                finding="No claim-specific authorization record was available through public web evidence.",
                reviewer_action="Check internal authorization, admission, and member benefit systems.",
            )
        )

    unresolved = [
        check
        for check in checks
        if check.status in {"Internal verification required", "Not publicly verifiable"}
    ]

    if unresolved:
        overall_status = "Internal Verification Required"
        summary = (
            "Clinical, coding, and general policy evidence can be evaluated from public sources, "
            "but payer, provider, or authorization checks require internal verification before final claim action."
        )
    else:
        overall_status = "Public Verification Completed"
        summary = (
            "Public evidence contained usable payer/provider verification signals. "
            "Internal confirmation is still recommended."
        )

    return VerificationResult(
        overall_status=overall_status,
        summary=summary,
        checks=checks,
    )