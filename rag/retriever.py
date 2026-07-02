"""Live online RAG retriever for ClinIQ.

This file performs live online evidence retrieval.

Main responsibilities:
1. Build claim-aware online search queries.
2. Use Query Planner queries when provided.
3. Search public web sources.
4. Fetch readable page text when possible.
5. Fall back to search snippets when sites block fetches.
6. Filter low-quality / irrelevant sources.
7. Prefer official, clinical, coding, payer, CMS, and policy sources.
8. Save retrieved evidence into online_knowledge/.
9. Add explicit verification gaps for payer, provider, and authorization evidence.

Important:
- This module does NOT use predefined local knowledge files.
- Scores are internal only.
- Scores are never displayed in the PDF or Streamlit tables.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger

try:
    from bs4 import XMLParsedAsHTMLWarning
    import warnings

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except Exception:
    pass

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None  # type: ignore

try:
    from ddgs import DDGS  # type: ignore
except Exception:
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except Exception:
        DDGS = None  # type: ignore

from models.decision import EvidenceItem, PatternAnalysis, RetrievedEvidence
from utils.config_loader import get_project_root, load_config


TOKEN_RE = re.compile(r"[a-zA-Z0-9.]+")


COLLECTION_LABELS = {
    "clinical": "Clinical Evidence",
    "policies": "Policy / Coverage Evidence",
    "coding": "Coding Evidence",
    "medical_necessity": "Medical Necessity Evidence",
    "historical": "Public Reference Evidence",
}


RETRIEVED_FOR = {
    "clinical": "Clinical support",
    "policies": "Policy / coverage",
    "coding": "Coding support",
    "medical_necessity": "Medical necessity",
    "historical": "Public reference",
}


STOPWORDS = {
    "the",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "without",
    "by",
    "from",
    "as",
    "at",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "case",
    "claim",
    "claims",
    "patient",
    "member",
    "provider",
    "payer",
    "plan",
    "policy",
    "code",
    "codes",
    "procedure",
    "procedures",
    "diagnosis",
    "diagnoses",
    "documentation",
    "document",
    "documents",
    "medical",
    "clinical",
    "service",
    "services",
    "report",
    "reports",
    "record",
    "records",
    "initial",
    "encounter",
    "unspecified",
    "right",
    "left",
    "other",
    "office",
    "visit",
    "hospital",
    "inpatient",
    "outpatient",
    "treatment",
    "evaluation",
    "management",
    "consult",
    "consultation",
    "history",
    "exam",
    "review",
    "support",
    "supports",
    "supporting",
    "national",
    "health",
    "standard",
    "general",
    "metro",
}


# Hard blocked sources. These should never appear in the report.
BLOCKED_DOMAINS = {
    "brainly.com",
    "brainly.in",
    "quizlet.com",
    "coursehero.com",
    "chegg.com",
    "studocu.com",
    "scribd.com",
    "slideshare.net",
    "answers.com",
    "quora.com",
    "reddit.com",
    "medium.com",
    "vocal.media",
    "pinterest.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
}


# Weak domains are not fully blocked, but they get penalized.
WEAK_DOMAINS = {
    "pabau.com",
    "legalclarity.org",
    "medicalcoding.online",
    "revenuees.com",
    "prombs.com",
    "codingclarified.com",
    "mdclarity.com",
    "icdcodes.ai",
    "icdlist.com",
    "checkicd10.com",
    "outsourcestrategies.com",
}


TRUSTED_DOMAINS = {
    "clinical": {
        "nih.gov",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "ahrq.gov",
        "cdc.gov",
        "msdmanuals.com",
        "aafp.org",
        "heart.org",
        "stroke.org",
        "acc.org",
        "escardio.org",
    },
    "policies": {
        "cms.gov",
        "medicare.gov",
        "uhcprovider.com",
        "aetna.com",
        "cigna.com",
        "anthem.com",
        "bcbs",
        "bluecross",
        "blueshield",
        "humana.com",
    },
    "coding": {
        "cms.gov",
        "icd10data.com",
        "aapc.com",
        "ama-assn.org",
        "findacode.com",
        "unboundmedicine.com",
    },
    "medical_necessity": {
        "cms.gov",
        "medicare.gov",
        "nih.gov",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "ahrq.gov",
        "aetna.com",
        "uhcprovider.com",
        "cigna.com",
    },
    "historical": {
        "nih.gov",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "cms.gov",
    },
}


POLICY_LANGUAGE = {
    "coverage",
    "medical policy",
    "clinical policy",
    "policy",
    "prior authorization",
    "preauthorization",
    "authorization",
    "medical necessity",
    "provider manual",
    "reimbursement policy",
    "benefit",
    "benefits",
    "covered",
    "not covered",
    "coverage criteria",
    "utilization management",
}


CODING_LANGUAGE = {
    "icd",
    "icd-10",
    "cpt",
    "hcpcs",
    "billing",
    "coding",
    "code",
    "diagnosis code",
    "procedure code",
}


MEDICAL_NECESSITY_LANGUAGE = {
    "medical necessity",
    "documentation",
    "clinical documentation",
    "criteria",
    "supporting records",
    "medical records",
    "operative report",
    "imaging",
    "history and physical",
    "severity of illness",
    "intensity of service",
}


HISTORICAL_LANGUAGE = {
    "case report",
    "case study",
    "example",
    "review",
    "documentation",
    "medical necessity",
}


@dataclass
class ClaimProfile:
    case_id: str = ""

    payer_name: str = ""
    payer_id: str = ""
    plan_type: str = ""
    policy_id: str = ""

    provider_name: str = ""
    provider_npi: str = ""
    provider_specialty: str = ""

    encounter_type: str = ""
    place_of_service: str = ""
    admission_date: str = ""
    discharge_date: str = ""

    claim_id: str = ""
    claim_type: str = ""
    total_billed: str = ""
    service_from: str = ""
    service_to: str = ""

    diagnosis_codes: list[str] = field(default_factory=list)
    diagnosis_descriptions: list[str] = field(default_factory=list)

    procedure_codes: list[str] = field(default_factory=list)
    procedure_descriptions: list[str] = field(default_factory=list)

    line_procedure_codes: list[str] = field(default_factory=list)
    line_diagnosis_pointers: list[str] = field(default_factory=list)
    line_billed_amounts: list[str] = field(default_factory=list)

    document_types: list[str] = field(default_factory=list)
    document_titles: list[str] = field(default_factory=list)
    document_summaries: list[str] = field(default_factory=list)

    pattern_text: str = ""
    raw_case_text: str = ""

    clinical_anchors: set[str] = field(default_factory=set)
    code_anchors: set[str] = field(default_factory=set)
    claim_anchors: set[str] = field(default_factory=set)
    document_anchors: set[str] = field(default_factory=set)

    def all_text(self) -> str:
        values: list[Any] = [
            self.case_id,
            self.payer_name,
            self.payer_id,
            self.plan_type,
            self.policy_id,
            self.provider_name,
            self.provider_npi,
            self.provider_specialty,
            self.encounter_type,
            self.place_of_service,
            self.admission_date,
            self.discharge_date,
            self.claim_id,
            self.claim_type,
            self.total_billed,
            self.service_from,
            self.service_to,
            *self.diagnosis_codes,
            *self.diagnosis_descriptions,
            *self.procedure_codes,
            *self.procedure_descriptions,
            *self.line_procedure_codes,
            *self.line_diagnosis_pointers,
            *self.line_billed_amounts,
            *self.document_types,
            *self.document_titles,
            *self.document_summaries,
            self.pattern_text,
        ]
        return " ".join(str(value) for value in values if value)


@dataclass
class OnlineDocument:
    category: str
    title: str
    url: str
    snippet: str
    text: str
    source_name: str
    cache_path: str | None = None
    internal_score: float = 0.0


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_join(values: list[str]) -> str:
    return " ".join(str(value) for value in values if value)


def _meaningful_terms(text: str) -> set[str]:
    terms: set[str] = set()

    for token in _tokens(text):
        if len(token) < 3:
            continue

        if token in STOPWORDS:
            continue

        if token.replace(".", "").isdigit():
            continue

        terms.add(token)

    return terms


def _flatten_json(value: Any) -> list[str]:
    values: list[str] = []

    if isinstance(value, dict):
        for child in value.values():
            values.extend(_flatten_json(child))
    elif isinstance(value, list):
        for item in value:
            values.extend(_flatten_json(item))
    elif value is not None:
        values.append(str(value))

    return values


def _code_variants(code: str) -> set[str]:
    cleaned = _norm(code)

    if not cleaned:
        return set()

    variants = {cleaned}

    if "." in cleaned:
        variants.add(cleaned.split(".")[0])

    return variants


def _contains_term(text: str, term: str) -> bool:
    cleaned = _norm(term)

    if not cleaned:
        return False

    pattern = r"(?<![a-zA-Z0-9])" + re.escape(cleaned) + r"(?![a-zA-Z0-9])"
    return re.search(pattern, text.lower()) is not None


def _contains_any(text: str, phrases: set[str]) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")


def _is_blocked_domain(url: str) -> bool:
    domain = _domain(url)

    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


def _is_weak_domain(url: str) -> bool:
    domain = _domain(url)

    return any(weak in domain for weak in WEAK_DOMAINS)


def _trusted_bonus(category: str, url: str) -> float:
    domain = _domain(url)

    for trusted in TRUSTED_DOMAINS.get(category, set()):
        if trusted in domain:
            return 22.0

    return 0.0


def _pattern_text(pattern: PatternAnalysis | None) -> str:
    if not pattern:
        return ""

    return " ".join(
        [
            pattern.summary,
            " ".join(pattern.relationships),
            " ".join(pattern.missing_information),
            " ".join(pattern.clinical_patterns),
            " ".join(pattern.alignment_checks),
            " ".join(item.message for item in pattern.inconsistencies),
        ]
    )


def _extract_profile(case: dict[str, Any], pattern: PatternAnalysis | None) -> ClaimProfile:
    case_info = case.get("case_information") or {}
    payer = case.get("payer") or {}
    provider = case.get("provider") or {}
    encounter = case.get("encounter") or {}
    claim = case.get("claim") or {}

    diagnoses = case.get("diagnoses") or []
    procedures = case.get("procedures") or []
    documents = case.get("supporting_documents") or []
    claim_lines = claim.get("lines") or []

    profile = ClaimProfile(
        case_id=str(case_info.get("case_id", "")),
        payer_name=str(payer.get("name", "")),
        payer_id=str(payer.get("payer_id", "")),
        plan_type=str(payer.get("plan_type", "")),
        policy_id=str(payer.get("policy_id", "")),
        provider_name=str(provider.get("name", "")),
        provider_npi=str(provider.get("npi", "")),
        provider_specialty=str(provider.get("specialty", "")),
        encounter_type=str(encounter.get("encounter_type", "")),
        place_of_service=str(encounter.get("place_of_service", "")),
        admission_date=str(encounter.get("admission_date", "")),
        discharge_date=str(encounter.get("discharge_date", "")),
        claim_id=str(claim.get("claim_id", "")),
        claim_type=str(claim.get("claim_type", "")),
        total_billed=str(claim.get("total_billed", "")),
        service_from=str(claim.get("service_from", "")),
        service_to=str(claim.get("service_to", "")),
        diagnosis_codes=[str(item.get("code", "")) for item in diagnoses],
        diagnosis_descriptions=[str(item.get("description", "")) for item in diagnoses],
        procedure_codes=[str(item.get("code", "")) for item in procedures],
        procedure_descriptions=[str(item.get("description", "")) for item in procedures],
        document_types=[str(item.get("document_type", "")) for item in documents],
        document_titles=[str(item.get("title", "")) for item in documents],
        document_summaries=[str(item.get("summary", "")) for item in documents],
        pattern_text=_pattern_text(pattern),
        raw_case_text=" ".join(_flatten_json(case)),
    )

    for line in claim_lines:
        procedure_code = line.get("procedure_code")

        if procedure_code:
            code = str(procedure_code)
            profile.line_procedure_codes.append(code)

            if code not in profile.procedure_codes:
                profile.procedure_codes.append(code)

        for pointer in line.get("diagnosis_pointers", []) or []:
            profile.line_diagnosis_pointers.append(str(pointer))

        if line.get("billed_amount") is not None:
            profile.line_billed_amounts.append(str(line.get("billed_amount")))

    clinical_text = _safe_join(profile.diagnosis_descriptions + profile.procedure_descriptions)
    profile.clinical_anchors = _meaningful_terms(clinical_text)

    for code in profile.diagnosis_codes + profile.procedure_codes + profile.line_procedure_codes:
        profile.code_anchors.update(_code_variants(code))

    claim_text = _safe_join(
        [
            profile.payer_name,
            profile.payer_id,
            profile.plan_type,
            profile.policy_id,
            profile.provider_name,
            profile.provider_npi,
            profile.provider_specialty,
            profile.encounter_type,
            profile.place_of_service,
            profile.claim_type,
            profile.claim_id,
            profile.total_billed,
        ]
    )
    profile.claim_anchors = _meaningful_terms(claim_text)

    doc_text = _safe_join(
        profile.document_types + profile.document_titles + profile.document_summaries
    )
    profile.document_anchors = _meaningful_terms(doc_text)

    return profile


def _query_for_category(profile: ClaimProfile, category: str) -> list[str]:
    diagnosis = _safe_join(profile.diagnosis_codes + profile.diagnosis_descriptions)
    procedure = _safe_join(profile.procedure_codes + profile.procedure_descriptions)
    docs = _safe_join(profile.document_types + profile.document_titles + profile.document_summaries)

    payer = _safe_join(
        [
            profile.payer_name,
            profile.payer_id,
            profile.plan_type,
            profile.policy_id,
        ]
    )

    provider = _safe_join(
        [
            profile.provider_name,
            profile.provider_npi,
            profile.provider_specialty,
        ]
    )

    encounter = _safe_join(
        [
            profile.claim_type,
            profile.encounter_type,
            f"place of service {profile.place_of_service}",
            f"admission {profile.admission_date}",
            f"discharge {profile.discharge_date}",
            f"service date {profile.service_from} {profile.service_to}",
        ]
    )

    line_context = _safe_join(
        [
            f"claim line codes {_safe_join(profile.line_procedure_codes)}",
            f"diagnosis pointers {_safe_join(profile.line_diagnosis_pointers)}",
            f"billed amount {profile.total_billed}",
        ]
    )

    if category == "clinical":
        return [
            f"{diagnosis} {procedure} clinical guideline standard of care",
            f"{diagnosis} {procedure} treatment pathway clinical documentation",
        ]

    if category == "policies":
        return [
            f"{payer} {procedure} coverage policy prior authorization medical necessity",
            f"{profile.payer_name} {profile.policy_id} {profile.plan_type} provider manual medical policy",
            f"{procedure} {diagnosis} insurance coverage policy authorization",
            f"{provider} provider NPI hospital facility verification",
        ]

    if category == "coding":
        return [
            f"{diagnosis} {procedure} ICD-10 CPT HCPCS coding reference",
            f"{line_context} diagnosis pointer coding claim line billing rules",
        ]

    if category == "medical_necessity":
        return [
            f"{diagnosis} {procedure} {docs} medical necessity documentation requirements",
            f"{encounter} {procedure} supporting records documentation clinical necessity",
        ]

    if category == "historical":
        return [
            f"{diagnosis} {procedure} public case report medical necessity documentation",
            f"{diagnosis} {procedure} example claim review coding documentation",
        ]

    return [profile.all_text()]


def _source_name(title: str, url: str) -> str:
    domain = _domain(url)
    cleaned_title = " ".join((title or "").split())[:85]

    if cleaned_title:
        return f"{cleaned_title} ({domain})"

    return domain or url[:100]


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or " ").strip()


def _extract_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
        tag.decompose()

    parts: list[str] = []

    for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "td"]):
        text = _clean_text(tag.get_text(" "))

        if len(text) >= 30:
            parts.append(text)

    return _clean_text(" ".join(parts))


def _search_web(query: str, max_results: int) -> list[dict[str, str]]:
    if DDGS is None:
        raise RuntimeError("Online search package is missing. Run: pip install ddgs")

    results: list[dict[str, str]] = []

    with DDGS() as searcher:  # type: ignore[operator]
        for result in searcher.text(query, max_results=max_results):
            href = result.get("href") or result.get("url") or ""

            if not href.startswith("http"):
                continue

            if _is_blocked_domain(href):
                logger.info(f"Blocked low-quality domain: {href}")
                continue

            results.append(
                {
                    "title": result.get("title", ""),
                    "url": href,
                    "snippet": result.get("body", "") or result.get("snippet", ""),
                }
            )

    return results


def _fetch_url(url: str, timeout: int, user_agent: str) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": user_agent},
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if "html" not in content_type and "text" not in content_type:
        return ""

    return _extract_page_text(response.text)


def _save_online_document(cache_dir: Path, doc: OnlineDocument) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = cache_dir / f"{_hash_url(doc.url)}.txt"

    payload = {
        "category": doc.category,
        "title": doc.title,
        "url": doc.url,
        "source_name": doc.source_name,
        "snippet": doc.snippet,
        "fetched_at_epoch": int(time.time()),
    }

    text = "---METADATA---\n"
    text += json.dumps(payload, indent=2)
    text += "\n---CONTENT---\n"
    text += doc.text

    path.write_text(text, encoding="utf-8")

    return str(path)


def _count_anchor_matches(text: str, anchors: set[str]) -> int:
    return sum(1 for anchor in anchors if _contains_term(text, anchor))


def _bm25_boost(query: str, docs: list[OnlineDocument]) -> dict[str, float]:
    if not docs:
        return {}

    query_tokens = _tokens(query)

    if not query_tokens:
        return {doc.url: 0.0 for doc in docs}

    corpus = [_tokens(doc.title + " " + doc.snippet + " " + doc.text[:3000]) for doc in docs]

    if BM25Okapi is not None:
        bm25 = BM25Okapi(corpus)
        raw_scores = bm25.get_scores(query_tokens)
        max_score = max(raw_scores) if len(raw_scores) else 0.0

        return {
            doc.url: (float(score) / float(max_score) * 100.0) if max_score > 0 else 0.0
            for doc, score in zip(docs, raw_scores)
        }

    query_set = set(query_tokens)
    raw_scores = [len(query_set.intersection(set(tokens))) for tokens in corpus]
    max_score = max(raw_scores) if raw_scores else 0

    return {
        doc.url: (float(score) / float(max_score) * 100.0) if max_score > 0 else 0.0
        for doc, score in zip(docs, raw_scores)
    }


def _category_language_bonus(category: str, text: str) -> float:
    text_lower = text.lower()

    if category == "policies":
        if not _contains_any(text_lower, POLICY_LANGUAGE):
            return -40.0

        return 22.0

    if category == "coding":
        if _contains_any(text_lower, CODING_LANGUAGE):
            return 22.0

        return -20.0

    if category == "medical_necessity":
        if _contains_any(text_lower, MEDICAL_NECESSITY_LANGUAGE):
            return 22.0

        return -12.0

    if category == "historical":
        if _contains_any(text_lower, HISTORICAL_LANGUAGE):
            return 12.0

        return -10.0

    return 0.0

def _rank_documents(
    category: str,
    docs: list[OnlineDocument],
    profile: ClaimProfile,
    query_text: str,
) -> list[OnlineDocument]:
    bm25_scores = _bm25_boost(query_text, docs)
    ranked: list[OnlineDocument] = []

    bad_source_terms = [
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

    for doc in docs:
        if _is_blocked_domain(doc.url):
            continue

        text = (doc.title + " " + doc.snippet + " " + doc.text).lower()

        if any(term in text for term in bad_source_terms):
            continue

        code_matches = _count_anchor_matches(text, profile.code_anchors)
        clinical_matches = _count_anchor_matches(text, profile.clinical_anchors)
        claim_matches = _count_anchor_matches(text, profile.claim_anchors)
        document_matches = _count_anchor_matches(text, profile.document_anchors)

        score = 0.0
        score += min(45.0, code_matches * 18.0)
        score += min(35.0, clinical_matches * 9.0)
        score += min(25.0, document_matches * 7.0)
        score += min(20.0, claim_matches * 5.0)
        score += 0.20 * bm25_scores.get(doc.url, 0.0)
        score += _trusted_bonus(category, doc.url)
        score += _category_language_bonus(category, text)

        if _is_weak_domain(doc.url):
            score -= 18.0

        if category == "clinical":
            if code_matches == 0 and clinical_matches == 0:
                continue

            if _contains_any(text, CODING_LANGUAGE) and not any(
                clinical_word in text
                for clinical_word in [
                    "guideline",
                    "clinical",
                    "standard of care",
                    "treatment",
                    "management",
                    "diagnosis",
                    "symptoms",
                    "evaluation",
                ]
            ):
                score -= 20.0

        if category == "policies":
            domain = _domain(doc.url)
            is_trusted_policy = any(trusted in domain for trusted in TRUSTED_DOMAINS["policies"])
            has_policy_language = _contains_any(text, POLICY_LANGUAGE)

            if not has_policy_language and not is_trusted_policy:
                continue

            if _contains_any(text, CODING_LANGUAGE) and not has_policy_language:
                continue

            if "icd diagnosis" in text or "icd-10 code" in text:
                score -= 25.0

        if category == "coding":
            if code_matches == 0 and not _contains_any(text, CODING_LANGUAGE):
                continue

        if category == "medical_necessity":
            if code_matches == 0 and clinical_matches == 0 and document_matches == 0:
                continue

            if not _contains_any(text, MEDICAL_NECESSITY_LANGUAGE):
                score -= 15.0

            if any(
                bad in text
                for bad in [
                    "software",
                    "emr",
                    "ehr",
                    "chronic rhinitis",
                    "pediatric",
                    "marketing",
                ]
            ):
                continue

        if score < 25.0:
            continue

        doc.internal_score = score
        ranked.append(doc)

    return sorted(ranked, key=lambda item: item.internal_score, reverse=True)


class HybridRetriever:
    """Live online evidence retriever.

    The class name stays HybridRetriever so app.workflow does not need to change.
    """

    CATEGORIES = ["clinical", "policies", "coding", "medical_necessity"]

    def __init__(self) -> None:
        self.config = load_config()
        self.web_cfg = self.config.get("web_retrieval", {})
        self.cache_root = get_project_root() / self.web_cfg.get("cache_directory", "online_knowledge")
        self.max_results = int(self.web_cfg.get("max_search_results_per_query", 5))
        self.max_docs = int(self.web_cfg.get("max_documents_per_category", 3))
        self.timeout = int(self.web_cfg.get("request_timeout_seconds", 12))
        self.min_text = int(self.web_cfg.get("min_text_characters", 350))
        self.user_agent = self.web_cfg.get("user_agent", "ClinIQ academic evidence bot")

    def _retrieve_category(
        self,
        category: str,
        profile: ClaimProfile,
        category_queries: list[str] | None = None,
    ) -> list[OnlineDocument]:
        docs_by_url: dict[str, OnlineDocument] = {}
        queries = category_queries if category_queries else _query_for_category(profile, category)

        for query in queries:
            try:
                logger.info(f"Online search [{category}]: {query}")
                results = _search_web(query, self.max_results)
            except Exception as exc:
                logger.warning(f"Online search failed for {category}: {exc}")
                continue

            for result in results:
                url = result["url"]

                if url in docs_by_url:
                    continue

                if _is_blocked_domain(url):
                    logger.info(f"Blocked low-quality domain: {url}")
                    continue

                title = _clean_text(result.get("title", ""))
                snippet = _clean_text(result.get("snippet", ""))
                text = ""

                try:
                    text = _fetch_url(url, self.timeout, self.user_agent)
                except Exception as exc:
                    logger.info(f"Fetch fallback: using search snippet only for {url} ({exc})")

                if len(text) < self.min_text:
                    text = snippet

                if len(text) < 80:
                    continue

                doc = OnlineDocument(
                    category=category,
                    title=title,
                    url=url,
                    snippet=snippet,
                    text=text[:10000],
                    source_name=_source_name(title, url),
                )

                doc.cache_path = _save_online_document(self.cache_root / category, doc)
                docs_by_url[url] = doc

        query_text = " ".join(queries)
        ranked = _rank_documents(category, list(docs_by_url.values()), profile, query_text)
        final_docs = ranked[: self.max_docs]

        if final_docs:
            logger.info(
                f"Retrieved {len(final_docs)} online item(s) for {category}: "
                + ", ".join(doc.source_name for doc in final_docs)
            )
        else:
            logger.warning(f"No relevant online evidence found for category={category}")

        return final_docs

    def _public_exact_match_found(self, evidence: RetrievedEvidence, search_terms: list[str]) -> bool:
        terms = [_norm(term) for term in search_terms if _norm(term)]

        if not terms:
            return False

        for item in evidence.all_items():
            if item.metadata.get("verification_gap"):
                continue

            combined = _norm(
                " ".join(
                    [
                        item.source,
                        item.content,
                        item.url or "",
                    ]
                )
            )

            for term in terms:
                if term and term in combined:
                    return True

        return False

    def _append_verification_gaps(self, evidence: RetrievedEvidence, profile: ClaimProfile) -> None:
        payer_terms = [
            profile.payer_name,
            profile.payer_id,
            profile.policy_id,
        ]

        provider_terms = [
            profile.provider_name,
            profile.provider_npi,
        ]

        payer_found = self._public_exact_match_found(evidence, payer_terms)
        provider_found = self._public_exact_match_found(evidence, provider_terms)

        if not payer_found:
            evidence.policies.append(
                EvidenceItem(
                    collection="Payer Policy Public Check",
                    source="No exact public payer policy source found",
                    retrieved_for="Payer verification",
                    content=(
                        f"No exact public online source was found for payer '{profile.payer_name}', "
                        f"plan '{profile.plan_type}', or policy ID '{profile.policy_id}'. "
                        "Reviewer should verify plan benefits, exclusions, coverage limitations, "
                        "and prior authorization rules in the payer/provider portal or internal policy system."
                    ),
                    url=None,
                    internal_score=0.0,
                    metadata={"verification_gap": True},
                )
            )

        if not provider_found:
            evidence.policies.append(
                EvidenceItem(
                    collection="Provider Public Check",
                    source="No exact public provider/NPI source found",
                    retrieved_for="Provider verification",
                    content=(
                        f"No exact public online source was found for provider '{profile.provider_name}' "
                        f"with NPI '{profile.provider_npi}'. Reviewer should verify facility status, "
                        "network status, billing/rendering provider role, and specialty alignment internally."
                    ),
                    url=None,
                    internal_score=0.0,
                    metadata={"verification_gap": True},
                )
            )

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

    def retrieve(
        self,
        case: dict[str, Any],
        pattern: PatternAnalysis | None,
        query_plan: dict[str, Any] | None = None,
    ) -> RetrievedEvidence:
        evidence = RetrievedEvidence()
        profile = _extract_profile(case, pattern)

        logger.info(
            "Online RAG claim profile: "
            f"case_id={profile.case_id}, "
            f"payer={profile.payer_name}, "
            f"plan={profile.plan_type}, "
            f"policy_id={profile.policy_id}, "
            f"provider={profile.provider_name}, "
            f"npi={profile.provider_npi}, "
            f"specialty={profile.provider_specialty}, "
            f"claim_type={profile.claim_type}, "
            f"pos={profile.place_of_service}, "
            f"diagnosis_codes={profile.diagnosis_codes}, "
            f"procedure_codes={profile.procedure_codes}, "
            f"clinical_anchors={sorted(profile.clinical_anchors)}, "
            f"code_anchors={sorted(profile.code_anchors)}"
        )

        planned_queries = {}
        if query_plan:
            planned_queries = query_plan.get("queries", {}) or {}

        target_map = {
            "clinical": evidence.clinical,
            "policies": evidence.policies,
            "coding": evidence.coding,
            "medical_necessity": evidence.medical_necessity,
            "historical": evidence.historical,
        }

        for category in self.CATEGORIES:
            category_queries = planned_queries.get(category)
            docs = self._retrieve_category(category, profile, category_queries=category_queries)

            for doc in docs:
                excerpt = _clean_text(doc.text[:1200])

                target_map[category].append(
                    EvidenceItem(
                        collection=COLLECTION_LABELS[category],
                        source=doc.source_name,
                        retrieved_for=RETRIEVED_FOR[category],
                        content=excerpt,
                        url=doc.url,
                        internal_score=doc.internal_score,
                        metadata={
                            "url": doc.url,
                            "cache_path": doc.cache_path,
                            "category": category,
                            "domain": _domain(doc.url),
                        },
                    )
                )

        #self._append_verification_gaps(evidence, profile)

        return evidence