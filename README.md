# ClinIQ

## Agentic Clinical Pattern Recognition and Decision Support Platform

ClinIQ is a hackathon proof-of-concept project built for the Cotiviti internship assessment under the topic:

**Clinical Decision Making and Pattern Recognition in Healthcare**

The project demonstrates how an agentic AI workflow can support healthcare claim review by combining structured claim analysis, clinical pattern recognition, online evidence retrieval, verification-aware reasoning, and reviewer-facing PDF reporting.

ClinIQ is designed as a **decision-support intelligence platform**, not an automated claim approval or denial system.

---

## Project Goal

Healthcare claim review often requires comparing diagnoses, procedures, documentation, coding references, payer policy, provider information, authorization status, and medical necessity evidence.

The goal of ClinIQ is to show how AI can help reviewers:

- Recognize clinical and coding patterns in structured claim data
- Retrieve public evidence using an online RAG pipeline
- Check evidence quality before reasoning
- Separate public evidence from internal verification requirements
- Generate transparent reviewer-oriented recommendations
- Produce a PDF Decision Intelligence Report

The system intentionally avoids final approve or deny language because real claim action often requires secure internal payer/provider data that is not publicly available.

---

## Important Disclaimer

ClinIQ is a proof of concept for decision-support only.

It does **not** approve, deny, adjudicate, or pay claims.

Public online evidence can support clinical, coding, policy, and medical necessity reasoning, but it cannot reliably verify:

- Payer-specific benefits
- Member eligibility
- Provider network status
- Authorization records
- Contract terms
- Internal payer policy rules

Because of this, ClinIQ explicitly reports **Internal Verification Required** when claim-specific verification cannot be completed using public information.

The final decision always remains with a qualified human reviewer.

---

## What ClinIQ Demonstrates

ClinIQ demonstrates the main concepts of the selected assessment topic:

| Topic Area | How ClinIQ Demonstrates It |
|---|---|
| Clinical Decision Making | Supports reviewer decisions using structured evidence |
| Pattern Recognition | Detects clinical, coding, documentation, and claim patterns |
| Chain Reasoning | Uses a staged workflow instead of one direct LLM prompt |
| Agentic Generative AI | Uses multiple specialized workflow agents |
| Classification | Categorizes evidence into clinical, policy, coding, and medical necessity |
| Inference | Identifies verification gaps and recommends next review action |
| Treatment | Analyzes diagnosis, procedure, and supporting clinical information |
| Payment | Supports payment integrity and claim review reasoning |
| Operations | Produces reviewer actions for payer, provider, and authorization checks |

Prediction, clustering, and time-series anomaly detection are treated as future extensions because they require historical longitudinal claim datasets.

---

## High-Level Workflow

```text
Structured Claim JSON
        ↓
Claim Validation
        ↓
Pattern Recognition Agent
        ↓
Query Planner Agent
        ↓
Online RAG Retriever
        ↓
Evidence Quality Agent
        ↓
Retry Retrieval Once If Evidence Is Weak
        ↓
Finalize Evidence Gaps
        ↓
Verification Agent
        ↓
Decision Intelligence Agent
        ↓
PDF Report Agent
