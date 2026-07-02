# ClinIQ

**Agentic Clinical Pattern Recognition and Decision Support Platform**

Version 1.1 redesign: **live online RAG** + concise reviewer report.

## What changed

- The app no longer depends on predefined local `knowledge/*.txt` files.
- When a claim is submitted, the RAG layer searches the web, fetches online pages/snippets, and stores the fetched evidence under `online_knowledge/`.
- The report is concise and includes only:
  - Claim snapshot
  - Patterns recognized
  - Online data retrieved: collection, source, retrieved-for
  - Decision-making summary
  - Evidence support
  - Recommendation
  - Reviewer actions
- No retrieval scores are shown.
- No confidence levels are shown.
- The system never approves or denies claims.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.12 is recommended for the AI ecosystem. Python 3.13 may work but can produce more package warnings.

## Run

```bash
streamlit run app/streamlit_app.py
```

## Online RAG behavior

The live RAG layer is in:

```text
rag/retriever.py
```

For each claim, it builds online search queries from:

- Diagnosis codes and descriptions
- Procedure codes and descriptions
- Claim lines and diagnosis pointers
- Payer, plan type, policy ID
- Provider, NPI, specialty
- Encounter type and place of service
- Admission/discharge/service dates
- Billed amount
- Supporting document summaries
- Pattern-agent findings

Fetched evidence is cached here:

```text
online_knowledge/
```

This folder is generated at runtime and is ignored by Git.

## Important limitation

The app cannot retrieve an exact payer policy, provider network record, or authorization record unless it is publicly available online. If exact evidence is not found, the Decision Agent should list it as an evidence gap instead of inventing it.
