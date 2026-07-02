"""LangGraph workflow orchestration for ClinIQ.

9/10 workflow:

Validate
→ Pattern Agent
→ Query Planner Agent
→ Online RAG Retriever
→ Evidence Quality Agent
→ Retry once if weak
→ Finalize evidence gaps
→ Verification Agent
→ Decision Agent
→ Report Agent
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from agents.decision_agent import generate_decision
from agents.evidence_quality_agent import (
    append_gap_items,
    build_retry_query_plan,
    merge_evidence,
    review_evidence,
)
from agents.pattern_agent import analyze_patterns
from agents.query_planner_agent import build_query_plan
from agents.report_agent import generate_report
from agents.verification_agent import run_verification
from models.decision import WorkflowState
from rag.retriever import HybridRetriever
from utils.json_validator import validate_and_normalize


class GraphState(TypedDict, total=False):
    case_id: str
    raw_case: dict[str, Any]
    normalized_case: dict[str, Any]
    pattern_analysis: Any
    query_plan: dict[str, Any]
    evidence: Any
    evidence_review: dict[str, Any]
    verification_result: Any
    decision: Any
    report_path: str | None
    retrieval_attempts: int
    workflow_steps: list[str]
    errors: list[str]
    override_model: str | None


def _append_step(state: GraphState, step: str) -> list[str]:
    steps = list(state.get("workflow_steps", []))
    steps.append(step)
    return steps


def validate_node(state: GraphState) -> GraphState:
    logger.info("Validating case JSON")

    try:
        case = validate_and_normalize(state["raw_case"])
        normalized = case.to_summary_dict()

        return {
            **state,
            "case_id": normalized["case_information"]["case_id"],
            "normalized_case": normalized,
            "workflow_steps": _append_step(state, "JSON validation complete"),
        }

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"Validation failed: {exc}")

        return {
            **state,
            "errors": errors,
            "workflow_steps": _append_step(state, "Validation failed"),
        }


def pattern_node(state: GraphState) -> GraphState:
    logger.info("Running pattern recognition")

    try:
        pattern = analyze_patterns(state["normalized_case"])

        return {
            **state,
            "pattern_analysis": pattern,
            "workflow_steps": _append_step(state, "Pattern recognition complete"),
        }

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"Pattern agent failed: {exc}")

        return {
            **state,
            "errors": errors,
            "workflow_steps": _append_step(state, "Pattern recognition failed"),
        }


def query_planner_node(state: GraphState) -> GraphState:
    logger.info("Planning online retrieval queries")

    try:
        query_plan = build_query_plan(
            state["normalized_case"],
            state.get("pattern_analysis"),
        )

        logger.info(
            "Query Planner generated categories: "
            + ", ".join(query_plan.get("queries", {}).keys())
        )

        return {
            **state,
            "query_plan": query_plan,
            "workflow_steps": _append_step(state, "Query planning complete"),
        }

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"Query Planner failed: {exc}")

        return {
            **state,
            "errors": errors,
            "workflow_steps": _append_step(state, "Query planning failed"),
        }


def retrieval_node(state: GraphState) -> GraphState:
    logger.info("Running live online evidence retrieval")

    try:
        retriever = HybridRetriever()

        evidence = retriever.retrieve(
            state["normalized_case"],
            state.get("pattern_analysis"),
            query_plan=state.get("query_plan"),
        )

        attempts = int(state.get("retrieval_attempts", 0)) + 1

        return {
            **state,
            "evidence": evidence,
            "retrieval_attempts": attempts,
            "workflow_steps": _append_step(
                state,
                f"Online evidence retrieval complete — attempt {attempts}",
            ),
        }

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"Retriever failed: {exc}")

        return {
            **state,
            "errors": errors,
            "workflow_steps": _append_step(state, "Online evidence retrieval failed"),
        }


def evidence_quality_node(state: GraphState) -> GraphState:
    logger.info("Reviewing evidence quality")

    try:
        review = review_evidence(
            state["normalized_case"],
            state["evidence"],
            retrieval_attempts=int(state.get("retrieval_attempts", 1)),
        )

        gap_count = len(review.get("gaps", []))

        return {
            **state,
            "evidence_review": review,
            "workflow_steps": _append_step(
                state,
                f"Evidence quality review complete — {gap_count} gap(s) found",
            ),
        }

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"Evidence Quality Agent failed: {exc}")

        return {
            **state,
            "errors": errors,
            "workflow_steps": _append_step(state, "Evidence quality review failed"),
        }


def retry_retrieval_node(state: GraphState) -> GraphState:
    logger.info("Retrying retrieval with improved query plan")

    try:
        retry_plan = build_retry_query_plan(
            state["normalized_case"],
            state.get("evidence_review", {}),
        )

        retriever = HybridRetriever()

        retry_evidence = retriever.retrieve(
            state["normalized_case"],
            state.get("pattern_analysis"),
            query_plan=retry_plan,
        )

        merged = merge_evidence(state["evidence"], retry_evidence)

        attempts = int(state.get("retrieval_attempts", 1)) + 1

        return {
            **state,
            "query_plan": retry_plan,
            "evidence": merged,
            "retrieval_attempts": attempts,
            "workflow_steps": _append_step(
                state,
                f"Agentic retrieval retry complete — attempt {attempts}",
            ),
        }

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"Retry retrieval failed: {exc}")

        return {
            **state,
            "errors": errors,
            "workflow_steps": _append_step(state, "Agentic retrieval retry failed"),
        }


def finalize_evidence_node(state: GraphState) -> GraphState:
    logger.info("Finalizing evidence gaps")

    try:
        final_evidence = append_gap_items(
            state["evidence"],
            state["normalized_case"],
            state.get("evidence_review", {}),
        )

        return {
            **state,
            "evidence": final_evidence,
            "workflow_steps": _append_step(state, "Evidence gaps finalized"),
        }

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"Evidence finalization failed: {exc}")

        return {
            **state,
            "errors": errors,
            "workflow_steps": _append_step(state, "Evidence finalization failed"),
        }


def verification_node(state: GraphState) -> GraphState:
    logger.info("Running payer/provider/authorization verification")

    try:
        verification_result = run_verification(
            state["normalized_case"],
            state["evidence"],
            state.get("evidence_review", {}),
        )

        return {
            **state,
            "verification_result": verification_result,
            "workflow_steps": _append_step(
                state,
                f"Verification complete — {verification_result.overall_status}",
            ),
        }

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"Verification Agent failed: {exc}")

        return {
            **state,
            "errors": errors,
            "workflow_steps": _append_step(state, "Verification failed"),
        }


def decision_node(state: GraphState) -> GraphState:
    logger.info("Running decision intelligence")

    try:
        decision = generate_decision(
            state["normalized_case"],
            state["pattern_analysis"],
            state["evidence"],
            verification_result=state.get("verification_result"),
            override_model=state.get("override_model"),
        )

        return {
            **state,
            "decision": decision,
            "workflow_steps": _append_step(state, "Decision intelligence complete"),
        }

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"Decision agent failed: {exc}")

        return {
            **state,
            "errors": errors,
            "workflow_steps": _append_step(state, "Decision intelligence failed"),
        }


def report_node(state: GraphState) -> GraphState:
    logger.info("Generating report")

    try:
        report_path = generate_report(
            state["normalized_case"],
            state["pattern_analysis"],
            state["evidence"],
            state["decision"],
            verification_result=state.get("verification_result"),
        )

        return {
            **state,
            "report_path": str(report_path),
            "workflow_steps": _append_step(state, "Report generation complete"),
        }

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"Report agent failed: {exc}")

        return {
            **state,
            "errors": errors,
            "workflow_steps": _append_step(state, "Report generation failed"),
        }


def should_retry_retrieval(state: GraphState) -> str:
    review = state.get("evidence_review") or {}
    attempts = int(state.get("retrieval_attempts", 1))

    if review.get("retry_needed") and attempts < 2:
        return "retry"

    return "finalize"


def build_workflow():
    graph = StateGraph(GraphState)

    graph.add_node("validate", validate_node)
    graph.add_node("pattern", pattern_node)
    graph.add_node("query_planner", query_planner_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("evidence_quality", evidence_quality_node)
    graph.add_node("retry_retrieval", retry_retrieval_node)
    graph.add_node("finalize_evidence", finalize_evidence_node)
    graph.add_node("verification", verification_node)
    graph.add_node("decision", decision_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("validate")

    graph.add_edge("validate", "pattern")
    graph.add_edge("pattern", "query_planner")
    graph.add_edge("query_planner", "retrieve")
    graph.add_edge("retrieve", "evidence_quality")

    graph.add_conditional_edges(
        "evidence_quality",
        should_retry_retrieval,
        {
            "retry": "retry_retrieval",
            "finalize": "finalize_evidence",
        },
    )

    graph.add_edge("retry_retrieval", "evidence_quality")
    graph.add_edge("finalize_evidence", "verification")
    graph.add_edge("verification", "decision")
    graph.add_edge("decision", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_analysis(raw_case: dict[str, Any], override_model: str | None = None) -> WorkflowState:
    workflow = build_workflow()

    result = workflow.invoke(
        {
            "raw_case": raw_case,
            "workflow_steps": ["JSON loaded"],
            "errors": [],
            "retrieval_attempts": 0,
            "override_model": override_model,
        }
    )

    return WorkflowState.model_validate(result)