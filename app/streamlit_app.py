"""Streamlit frontend for ClinIQ."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.workflow import run_analysis
from demo_cases.generate_demo_cases import write_demo_cases
from utils.config_loader import load_config
from utils.logging_setup import setup_logging


setup_logging()
write_demo_cases(PROJECT_ROOT / "demo_cases")
config = load_config()

st.set_page_config(
    page_title="ClinIQ Evidence Analysis",
    page_icon="🏥",
    layout="wide",
)


def _safe(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    clean = [{str(k): _safe(v) for k, v in row.items()} for row in rows]
    frame = pd.DataFrame(clean)
    for column in frame.columns:
        frame[column] = frame[column].astype(str)
    return frame


def _table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No data available.")
        return
    st.dataframe(_df(rows), width="stretch", hide_index=True)


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "N/A"


def _claim_snapshot(case_payload: dict[str, Any]) -> None:
    payer = case_payload.get("payer") or {}
    provider = case_payload.get("provider") or {}
    encounter = case_payload.get("encounter") or {}
    claim = case_payload.get("claim") or {}
    rows = [
        {"Claim Aspect": "Payer / Plan", "Value": f"{_safe(payer.get('name'))} / {_safe(payer.get('plan_type'))}"},
        {"Claim Aspect": "Policy ID", "Value": payer.get("policy_id")},
        {"Claim Aspect": "Provider / NPI", "Value": f"{_safe(provider.get('name'))} / {_safe(provider.get('npi'))}"},
        {"Claim Aspect": "Provider Specialty", "Value": provider.get("specialty")},
        {"Claim Aspect": "Claim Type / POS", "Value": f"{_safe(claim.get('claim_type'))} / {_safe(encounter.get('place_of_service'))}"},
        {"Claim Aspect": "Dates", "Value": f"{_safe(claim.get('service_from'))} to {_safe(claim.get('service_to'))}"},
        {"Claim Aspect": "Total Billed", "Value": _money(claim.get("total_billed"))},
    ]
    _table(rows)


def _evidence_table(result: Any) -> None:
    rows = []
    for item in result.evidence.all_items():
        rows.append(
            {
                "Collection Category": item.collection,
                "Source Retrieved": item.source,
                "Retrieved For": item.retrieved_for,
            }
        )
    _table(rows)


st.title("ClinIQ")
st.caption("Agentic Clinical Pattern Recognition and Decision Support Platform")

with st.expander("Important runtime note", expanded=False):
    st.write(
        "This version does not use predefined local knowledge files. When you run a claim, the RAG layer searches the web, fetches online pages/snippets, and stores them under `online_knowledge/` for that run. If internet search is blocked or the `ddgs` package is missing, retrieval will return evidence gaps instead of inventing facts."
    )

col1, col2 = st.columns([1, 1])

with col1:
    demo_files = sorted((PROJECT_ROOT / "demo_cases").glob("*.json"))
    demo_names = [path.stem.replace("_", " ").title() for path in demo_files]
    selected_demo = st.selectbox("Select Demo Case", ["None"] + demo_names)

with col2:
    llm_options = {
        "Groq Llama 3.3 70B": "groq/llama-3.3-70b-versatile",
        "Gemini 2.0 Flash": "gemini/gemini-2.0-flash",
        "Ollama Llama 3.1 8B": "ollama/llama3.1:8b",
    }
    selected_llm = st.selectbox("Choose LLM", list(llm_options.keys()))

uploaded = st.file_uploader("Or upload healthcare case JSON", type=["json"])
run_clicked = st.button("Run Online Evidence Analysis", type="primary")

if run_clicked:
    if uploaded is not None:
        case_payload = json.loads(uploaded.getvalue().decode("utf-8"))
    elif selected_demo != "None":
        demo_path = demo_files[demo_names.index(selected_demo)]
        case_payload = json.loads(demo_path.read_text(encoding="utf-8"))
    else:
        st.error("Select a demo case or upload JSON before running analysis.")
        st.stop()

    with st.spinner("Running pattern recognition, online retrieval, decision support, and PDF generation..."):
        result = run_analysis(case_payload, override_model=llm_options[selected_llm])

    st.success("Analysis complete")

    if result.errors:
        st.warning("Workflow warnings/errors")
        for error in result.errors:
            st.write(f"- {error}")

    st.subheader("Recommendation")
    if result.decision:
        st.info(result.decision.recommendation.value)
        st.write(result.decision.reasoning_summary)

    tabs = st.tabs(["Claim Snapshot", "Patterns", "Online Evidence", "Decision Support", "Report"])

    with tabs[0]:
        _claim_snapshot(case_payload)

    with tabs[1]:
        if result.pattern_analysis:
            for point in result.pattern_analysis.reviewer_points() or [result.pattern_analysis.summary]:
                st.write(f"- {point}")

    with tabs[2]:
        if result.evidence and result.evidence.all_items():
            _evidence_table(result)
            for item in result.evidence.all_items():
                with st.expander(f"{item.collection} — {item.source}"):
                    if item.url:
                        st.write(item.url)
                    st.write(item.content)
        else:
            st.info("No online evidence retrieved. Check internet/search package availability.")

    with tabs[3]:
        if result.decision:
            st.markdown("**How evidence supported the recommendation**")
            for item in result.decision.decision_support:
                st.write(f"- {item}")
            if result.decision.missing_information:
                st.markdown("**Evidence gaps / missing information**")
                for item in result.decision.missing_information:
                    st.write(f"- {item}")
            if result.decision.suggested_next_actions:
                st.markdown("**Suggested reviewer actions**")
                for item in result.decision.suggested_next_actions:
                    st.write(f"- {item}")

    with tabs[4]:
        if result.report_path and Path(result.report_path).exists():
            with Path(result.report_path).open("rb") as handle:
                st.download_button(
                    "Download Concise Decision Intelligence Report (PDF)",
                    data=handle.read(),
                    file_name=Path(result.report_path).name,
                    mime="application/pdf",
                )

st.divider()
st.markdown("**Disclaimer:** Decision-support only. Human reviewer retains final authority.")
