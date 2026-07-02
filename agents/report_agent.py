"""Concise professional PDF Report Generation Agent.

Report structure:
1. Claim Key Details
2. Patterns Recognized
3. Online Evidence Retrieved
4. Internal Verification Required
5. Decision-Making Summary
6. Clinical Evidence Position
7. Verification Status
8. Key Evidence Supporting the Recommendation
9. Final Recommendation
10. Suggested Reviewer Actions
11. Disclaimer

No confidence percentages.
No score columns.
No Purpose column.
Source names are clickable links.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from models.decision import (
    DecisionRecommendation,
    PatternAnalysis,
    RecommendationType,
    RetrievedEvidence,
    VerificationResult,
)
from utils.config_loader import get_project_root, load_config


BLUE = colors.HexColor("#1F4E79")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
BORDER = colors.HexColor("#CBD5E1")
TEXT_DARK = colors.HexColor("#111827")

MAX_TOTAL_EVIDENCE_ROWS = 7

CATEGORY_LIMITS = {
    "Clinical Evidence": 2,
    "Policy / Coverage Evidence": 2,
    "Coding Evidence": 2,
    "Medical Necessity Evidence": 1,
}

DROP_COLLECTIONS = {
    "Public Reference Evidence",
    "Public Reference / Similar Case Evidence",
}

WEAK_SOURCE_TERMS = [
    "emr software",
    "ehr software",
    "software",
    "chronic rhinitis",
    "pediatric",
    "ultimate guide",
    "billing services",
    "medical billing services",
    "reimbursement schedule",
    "brainly",
    "quizlet",
    "coursehero",
    "chegg",
    "reddit",
    "quora",
]


def _safe(value: Any) -> str:
    if value is None or value == "":
        return "N/A"

    if isinstance(value, float):
        return f"{value:,.2f}"

    return str(value)


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "N/A"


def _clean_text(value: Any, max_chars: int = 180) -> str:
    text = _safe(value).replace("\n", " ").strip()
    text = " ".join(text.split())

    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"

    return text


def _cell(value: Any, style: ParagraphStyle, max_chars: int = 180) -> Paragraph:
    return Paragraph(escape(_clean_text(value, max_chars=max_chars)), style)


def _link_cell(
    label: Any,
    url: str | None,
    style: ParagraphStyle,
    max_chars: int = 220,
) -> Paragraph:
    label_text = escape(_clean_text(label, max_chars=max_chars))

    if not url:
        return Paragraph(label_text, style)

    href = quoteattr(str(url))

    return Paragraph(
        f'<a href={href} color="#0B63CE"><u>{label_text}</u></a>',
        style,
    )


def _build_table(
    wrapped_rows: list[list[Paragraph]],
    widths: list[float],
    long: bool = False,
):
    table_cls = LongTable if long else Table

    table = table_cls(
        wrapped_rows,
        colWidths=widths,
        repeatRows=1,
        splitByRow=1,
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return table


def _table(rows: list[list[Any]], widths: list[float], styles, long: bool = False):
    wrapped_rows: list[list[Paragraph]] = []

    for row_index, row in enumerate(rows):
        wrapped_row: list[Paragraph] = []

        for col_index, value in enumerate(row):
            style = styles["TableHeader"] if row_index == 0 else styles["TableCell"]
            max_chars = 80 if col_index == 0 else 220
            wrapped_row.append(_cell(value, style, max_chars=max_chars))

        wrapped_rows.append(wrapped_row)

    return _build_table(wrapped_rows, widths, long=long)


def _section(title: str, body: str, styles) -> list:
    safe_body = escape(body or "N/A").replace("\n", "<br/>")

    return [
        Paragraph(escape(title), styles["SectionTitle"]),
        Spacer(1, 0.04 * inch),
        Paragraph(safe_body, styles["BodyText"]),
        Spacer(1, 0.09 * inch),
    ]


def _bullet_section(title: str, items: list[str], styles, max_items: int = 4) -> list:
    story: list = [
        Paragraph(escape(title), styles["SectionTitle"]),
        Spacer(1, 0.04 * inch),
    ]

    if not items:
        story.append(Paragraph("N/A", styles["BodyText"]))
    else:
        for item in items[:max_items]:
            story.append(
                Paragraph(
                    f"• {escape(_clean_text(item, max_chars=270))}",
                    styles["BodyText"],
                )
            )

    story.append(Spacer(1, 0.09 * inch))
    return story


def _claim_snapshot(case: dict[str, Any], styles) -> Table:
    payer = case.get("payer") or {}
    provider = case.get("provider") or {}
    encounter = case.get("encounter") or {}
    claim = case.get("claim") or {}

    rows = [
        ["Claim Detail", "Value"],
        ["Payer / Plan", f"{_safe(payer.get('name'))} / {_safe(payer.get('plan_type'))}"],
        ["Policy ID", _safe(payer.get("policy_id"))],
        ["Provider / NPI", f"{_safe(provider.get('name'))} / {_safe(provider.get('npi'))}"],
        [
            "Claim Type / POS",
            f"{_safe(claim.get('claim_type'))} / {_safe(encounter.get('place_of_service'))}",
        ],
        ["Service Dates", f"{_safe(claim.get('service_from'))} to {_safe(claim.get('service_to'))}"],
        ["Total Billed", _money(claim.get("total_billed"))],
    ]

    return _table(rows, [1.60 * inch, 5.10 * inch], styles)


def _patterns(pattern: PatternAnalysis) -> list[str]:
    try:
        points = pattern.reviewer_points()
    except Exception:
        points = []

    if points:
        return points[:5]

    return [pattern.summary]


def _is_gap_item(item: Any) -> bool:
    return bool((item.metadata or {}).get("verification_gap"))


def _is_weak_source(item: Any) -> bool:
    text = f"{item.source} {item.content}".lower()
    return any(term in text for term in WEAK_SOURCE_TERMS)


def _reportable_evidence_items(evidence: RetrievedEvidence):
    selected = []
    category_counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()

    sorted_items = sorted(
        evidence.all_items(),
        key=lambda item: getattr(item, "internal_score", 0.0),
        reverse=True,
    )

    for item in sorted_items:
        if _is_gap_item(item):
            continue

        if item.collection in DROP_COLLECTIONS:
            continue

        if item.collection not in CATEGORY_LIMITS:
            continue

        if _is_weak_source(item):
            continue

        key = (item.collection, item.source)

        if key in seen:
            continue

        current_count = category_counts.get(item.collection, 0)
        allowed_count = CATEGORY_LIMITS[item.collection]

        if current_count >= allowed_count:
            continue

        selected.append(item)
        seen.add(key)
        category_counts[item.collection] = current_count + 1

        if len(selected) >= MAX_TOTAL_EVIDENCE_ROWS:
            break

    return selected


def _evidence_table(evidence: RetrievedEvidence, styles) -> LongTable:
    wrapped_rows: list[list[Paragraph]] = [
        [
            _cell("Evidence Category", styles["TableHeader"]),
            _cell("Online Source Used", styles["TableHeader"]),
        ]
    ]

    selected_items = _reportable_evidence_items(evidence)

    if not selected_items:
        wrapped_rows.append(
            [
                _cell("No strong public source", styles["TableCell"]),
                _cell(
                    "No high-quality online source matched; manual review required.",
                    styles["TableCell"],
                ),
            ]
        )
    else:
        for item in selected_items:
            wrapped_rows.append(
                [
                    _cell(item.collection, styles["TableCell"], max_chars=80),
                    _link_cell(item.source, item.url, styles["TableCell"], max_chars=230),
                ]
            )

    return _build_table(
        wrapped_rows,
        [1.65 * inch, 5.05 * inch],
        long=True,
    )


def _verification_table(
    verification_result: VerificationResult | None,
    styles,
) -> LongTable | None:
    if verification_result is None or not verification_result.checks:
        return None

    wrapped_rows: list[list[Paragraph]] = [
        [
            _cell("Verification Check", styles["TableHeader"]),
            _cell("Status / Reviewer Action", styles["TableHeader"]),
        ]
    ]

    for check in verification_result.checks[:4]:
        wrapped_rows.append(
            [
                _cell(check.name, styles["TableCell"], max_chars=90),
                _cell(
                    f"{check.status}: {check.reviewer_action}",
                    styles["TableCell"],
                    max_chars=420,
                ),
            ]
        )

    return _build_table(
        wrapped_rows,
        [1.85 * inch, 4.85 * inch],
        long=True,
    )


def _decision_support_items(decision: DecisionRecommendation) -> list[str]:
    items = list(decision.decision_support or [])

    if not items and decision.reasoning_summary:
        items = [decision.reasoning_summary]

    return items[:4]


def _infer_clinical_position(decision: DecisionRecommendation) -> str:
    current = getattr(decision, "clinical_evidence_position", "Not Assessed")

    if current and current.lower() not in {"not assessed", "n/a", "none"}:
        return current

    recommendation = decision.recommendation
    text = f"{decision.reasoning_summary} {' '.join(decision.decision_support)}".lower()

    if recommendation == RecommendationType.LIKELY_SUPPORTED:
        return "Likely Supported"

    if recommendation == RecommendationType.ADDITIONAL_DOCUMENTATION:
        return "Additional Documentation Needed"

    if recommendation == RecommendationType.EVIDENCE_INSUFFICIENT:
        return "Evidence Insufficient"

    if recommendation == RecommendationType.LIKELY_NOT_SUPPORTED:
        return "Likely Not Supported"

    if recommendation == RecommendationType.REQUIRES_MANUAL_REVIEW:
        supported_phrases = [
            "clinical evidence supports",
            "evidence supports",
            "coding evidence supports",
            "medical necessity supports",
            "clinically aligned",
            "appears supported",
            "supports the use",
            "supports the diagnosis",
            "supports the treatment",
        ]

        if any(phrase in text for phrase in supported_phrases):
            return "Likely Supported"

        if "insufficient" in text:
            return "Evidence Insufficient"

        if "not supported" in text or "does not support" in text:
            return "Likely Not Supported"

        return "Likely Supported"

    return "Evidence Insufficient"


def _clinical_explanation(clinical_position: str) -> str:
    if clinical_position == "Likely Supported":
        return (
            "The available clinical and coding evidence appears to support the diagnosis-to-procedure "
            "relationship and the documented treatment pattern. This only reflects clinical evidence support "
            "and does not represent claim approval, payment approval, or final adjudication." 
            
        ) 

    if clinical_position == "Additional Documentation Needed":
        return (
            "Some clinical support exists, but the available documentation is not complete enough "
            "for a strong reviewer recommendation."
        )

    if clinical_position == "Evidence Insufficient":
        return (
            "The available evidence is not sufficient to determine whether the clinical or coding "
            "relationship is supported."
        )

    if clinical_position == "Likely Not Supported":
        return (
            "The available clinical or coding evidence does not clearly support the submitted "
            "diagnosis, procedure, or treatment pattern."
        )

    return (
        "The clinical evidence position could not be clearly determined from the available evidence."
    )

    story.append(Paragraph("Action Matrix", styles["SectionTitle"]))
    story.append(_action_matrix_table(decision, verification_result, styles))
    story.append(Spacer(1, 0.10 * inch))


def _verification_explanation(verification_status: str) -> str:
    if verification_status == "Public Verification Completed":
        return (
            "Public evidence contained usable verification signals. Internal confirmation is still "
            "recommended before final claim action."
        )

    if verification_status == "Not Publicly Verifiable":
        return (
            "Claim-specific payer policy, provider/NPI status, member benefits, and authorization "
            "records are not publicly available. These checks require internal payer or provider systems."
        )

    return (
        "Claim-specific payer policy, provider/NPI status, member benefits, and authorization "
        "records usually cannot be verified through public web evidence. These checks must be "
        "completed through internal payer, provider, or authorization systems before final claim action."
    )


def _final_recommendation_explanation(decision: DecisionRecommendation) -> str:
    recommendation = decision.recommendation

    if recommendation == RecommendationType.LIKELY_SUPPORTED:
        return (
            "The evidence appears aligned, but this remains decision-support only. A human reviewer "
            "retains final authority."
        )

    if recommendation == RecommendationType.REQUIRES_MANUAL_REVIEW:
        return (
            "The claim should not be automatically approved or denied. A human reviewer should verify "
            "payer, provider, authorization, and documentation details before final action."
        )

    if recommendation == RecommendationType.ADDITIONAL_DOCUMENTATION:
        return (
            "The claim may require additional records or clarification before a reviewer can make a "
            "final determination."
        )

    if recommendation == RecommendationType.EVIDENCE_INSUFFICIENT:
        return (
            "The available evidence is not strong enough to support a reviewer-facing recommendation."
        )

    if recommendation == RecommendationType.LIKELY_NOT_SUPPORTED:
        return (
            "The available evidence does not clearly support the submitted claim pattern. Human review "
            "is still required before any final action."
        )

    return "Human reviewer remains the final authority."

def _decision_support_items(decision: DecisionRecommendation) -> list[str]:
    items = list(decision.decision_support or [])

    if not items and decision.reasoning_summary:
        items = [decision.reasoning_summary]

    return items[:4]

def _action_matrix_table(
    decision: DecisionRecommendation,
    verification_result: VerificationResult | None,
    styles,
) -> Table:
    clinical_position = getattr(decision, "clinical_evidence_position", "Not Assessed")
    verification_status = getattr(
        decision,
        "verification_status",
        "Internal Verification Required",
    )

    if clinical_position == "Likely Supported":
        clinical_result = "Supported — Pending Internal Verification"
        coding_result = "Supported — Pending Internal Verification"
        medical_necessity_result = "Supported — Pending Internal Verification"
    elif clinical_position == "Additional Documentation Needed":
        clinical_result = "Additional Documentation Needed"
        coding_result = "Review Needed"
        medical_necessity_result = "Review Needed"
    elif clinical_position == "Evidence Insufficient":
        clinical_result = "Evidence Insufficient"
        coding_result = "Evidence Insufficient"
        medical_necessity_result = "Evidence Insufficient"
    elif clinical_position == "Likely Not Supported":
        clinical_result = "Not Clearly Supported"
        coding_result = "Not Clearly Supported"
        medical_necessity_result = "Not Clearly Supported"
    else:
        clinical_result = "Review Needed"
        coding_result = "Review Needed"
        medical_necessity_result = "Review Needed"

    payer_result = "Internal Verification Required"
    provider_result = "Internal Verification Required"
    authorization_result = "Internal Verification Required"

    if verification_result is not None:
        for check in verification_result.checks:
            name = check.name.lower()
            status = check.status

            if "payer" in name:
                payer_result = status
            elif "provider" in name or "hospital" in name:
                provider_result = status
            elif "authorization" in name:
                authorization_result = status

    rows = [
        ["Area", "Result"],
        ["Clinical Evidence", clinical_result],
        ["Coding", coding_result],
        ["Medical Necessity", medical_necessity_result],
        ["Payer Policy", payer_result],
        ["Provider Validation", provider_result],
        ["Authorization", authorization_result],
    ]

    return _table(rows, [2.25 * inch, 4.45 * inch], styles)


def generate_report(
    case: dict[str, Any],
    pattern: PatternAnalysis,
    evidence: RetrievedEvidence,
    decision: DecisionRecommendation,
    verification_result: VerificationResult | None = None,
):
    config = load_config().get("report", {})
    output_dir = get_project_root() / config.get("output_directory", "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    case_id = case.get("case_information", {}).get("case_id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ClinIQ_Report_{case_id}_{timestamp}.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.42 * inch,
        leftMargin=0.42 * inch,
        topMargin=0.42 * inch,
        bottomMargin=0.42 * inch,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=17,
            leading=20,
            textColor=BLUE,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontSize=11.8,
            leading=14,
            textColor=BLUE,
            spaceBefore=3,
            spaceAfter=3,
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.4,
            textColor=colors.white,
            wordWrap="CJK",
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontSize=7.3,
            leading=8.9,
            textColor=TEXT_DARK,
            wordWrap="CJK",
        )
    )

    styles.add(
        ParagraphStyle(
            name="FooterNote",
            parent=styles["BodyText"],
            fontSize=7.4,
            leading=9,
            textColor=colors.HexColor("#4B5563"),
        )
    )

    styles["BodyText"].fontSize = 8.8
    styles["BodyText"].leading = 11.2

    story: list = []

    report_title = config.get(
        "title",
        "ClinIQ: Agentic Clinical Pattern Recognition and Decision Support Platform",
    )

    story.append(Paragraph(escape(str(report_title)), styles["ReportTitle"]))
    story.append(
        Paragraph(
            f"Case ID: <b>{escape(str(case_id))}</b> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "Evidence-analysis support only — human reviewer retains final decision authority.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.10 * inch))

    story.append(Paragraph("Claim Key Details", styles["SectionTitle"]))
    story.append(_claim_snapshot(case, styles))
    story.append(Spacer(1, 0.10 * inch))

    story.extend(_bullet_section("Patterns Recognized", _patterns(pattern), styles, max_items=5))

    story.append(Paragraph("Online Evidence Retrieved", styles["SectionTitle"]))
    story.append(_evidence_table(evidence, styles))
    story.append(Spacer(1, 0.10 * inch))

    verification_table = _verification_table(verification_result, styles)
    if verification_table is not None:
        story.append(Paragraph("Internal Verification Required", styles["SectionTitle"]))

        if verification_result is not None:
            story.append(Paragraph(escape(verification_result.summary), styles["BodyText"]))
            story.append(Spacer(1, 0.04 * inch))

        story.append(verification_table)
        story.append(Spacer(1, 0.10 * inch))

    story.extend(_section("Evidence Analysis Summary", decision.reasoning_summary, styles))

    clinical_position = _infer_clinical_position(decision)
    clinical_explanation = _clinical_explanation(clinical_position)

    display_clinical_position = clinical_position

    if clinical_position == "Likely Supported":
        display_clinical_position = "Supportive Clinical Evidence Found — Pending Internal Verification"

    story.extend(
        _section(
            "Clinical Evidence Assessment",
            f"{display_clinical_position}\n\nExplanation: {clinical_explanation}",
            styles,
        )
    )

    verification_status = getattr(
        decision,
        "verification_status",
        "Internal Verification Required",
    )
    verification_explanation = _verification_explanation(verification_status)

    story.extend(
        _section(
            "Verification Status",
            f"{verification_status}\n\nExplanation: {verification_explanation}",
            styles,
        )
    )

    story.extend(
        _bullet_section(
            "Key Evidence Supporting the Recommendation",
            _decision_support_items(decision),
            styles,
            max_items=4,
        )
    )

    if getattr(decision, "final_recommendation_basis", ""):
        story.extend(
            _section(
                "Final Recommendation Basis",
                decision.final_recommendation_basis,
                styles,
            )
        )

    final_explanation = _final_recommendation_explanation(decision)

    recommended_next_action = decision.recommendation.value

    if decision.recommendation.value == "Requires Manual Review":
        next_action_explanation = (
            "Continue internal payer, provider, and authorization verification. If internal verification "
            "confirms payer coverage, authorization status, and provider eligibility, the claim may proceed "
            "through the organization's standard adjudication workflow."
        )
    else:
        next_action_explanation = final_explanation

    story.extend(
        _section(
            "Recommended Next Action",
            f"{recommended_next_action}\n\nExplanation: {next_action_explanation}",
            styles,
        )
    )

    if decision.suggested_next_actions:
        story.extend(
            _bullet_section(
                "Suggested Reviewer Actions",
                decision.suggested_next_actions,
                styles,
                max_items=3,
            )
        )

    story.append(Spacer(1, 0.05 * inch))
    story.append(
        Paragraph(
            "Disclaimer: This report provides decision intelligence support only. It does not approve, deny, or adjudicate claims.",
            styles["FooterNote"],
        )
    )

    doc.build(story)

    return output_path