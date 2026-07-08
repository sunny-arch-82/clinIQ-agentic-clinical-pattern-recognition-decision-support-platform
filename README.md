# ClinIQ

## Agentic Clinical Pattern Recognition and Decision Support Platform

ClinIQ is a hackathon proof-of-concept project built for the Cotiviti internship assessment under the topic:

**Clinical Decision Making and Pattern Recognition in Healthcare**

The project demonstrates how an agentic AI workflow can support healthcare claim review by combining structured claim analysis, clinical pattern recognition, online evidence retrieval, verification-aware reasoning, and reviewer-facing PDF reporting.

ClinIQ is designed as a **decision-support intelligence platform**, not an automated claim approval, denial, or adjudication system.

---

## Project Goal

Healthcare claim review often requires comparing diagnoses, procedures, supporting documentation, coding references, medical necessity evidence, payer policy, provider information, member eligibility, and authorization status.

The goal of ClinIQ is to show how AI can help reviewers:

- Recognize clinical and coding patterns in structured claim data
- Retrieve public evidence using an online RAG pipeline
- Check evidence quality before generating a recommendation
- Separate public evidence from internal verification requirements
- Generate transparent reviewer-oriented recommendations
- Produce a PDF Decision Intelligence Report

The system intentionally avoids final approve or deny language because real claim action often requires secure internal payer/provider data that is not publicly available.

---

## Important Disclaimer

ClinIQ is a proof of concept for decision support.

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
```

---

## Agentic Architecture

ClinIQ is not a single LLM prompt. The workflow is divided into specialized stages.

### 1. Pattern Recognition Agent

Detects clinical and claim-level patterns such as:

- Orthopedic procedure pattern
- Cardiac intervention pattern
- Respiratory treatment pattern
- Procedure documentation pattern
- Missing authorization/member benefit information

### 2. Query Planner Agent

Creates claim-specific retrieval queries for:

- Clinical evidence
- Policy/coverage evidence
- Coding references
- Medical necessity evidence

### 3. Online RAG Retriever

Retrieves public evidence from online sources and organizes it into evidence categories.

### 4. Evidence Quality Agent

Checks whether retrieved sources are relevant and useful. If evidence is weak, the workflow retries retrieval once.

### 5. Verification Agent

Separates publicly retrievable evidence from information that requires internal systems, such as payer policy, provider status, authorization records, and member eligibility.

### 6. Decision Intelligence Agent

Synthesizes the claim data, recognized patterns, retrieved evidence, and verification gaps into a reviewer-oriented recommendation.

### 7. Report Agent

Generates a concise PDF Decision Intelligence Report containing:

- Claim key details
- Patterns recognized
- Online evidence retrieved
- Internal verification required
- Evidence analysis summary
- Clinical evidence assessment
- Verification status
- Recommended next action
- Suggested reviewer actions

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| Workflow Orchestration | LangGraph |
| LLM Interface | LiteLLM / Groq-compatible setup |
| Data Validation | Pydantic |
| Retrieval | Online RAG pipeline |
| Report Generation | PDF generation utilities |
| Configuration | `.env.example`, config files |
| Demo Input | Structured healthcare claim JSON |

---

## Repository Structure

```text
ClinIQ/
├── agents/
│   ├── pattern_agent.py
│   ├── decision_agent.py
│   ├── verification_agent.py
│   └── report_agent.py
│
├── app/
│   ├── streamlit_app.py
│   └── workflow.py
│
├── config/
│   └── config files
│
├── demo_cases/
│   └── sample structured claim JSON cases
│
├── knowledge/
│   └── sample knowledge assets
│
├── models/
│   └── Pydantic models and workflow state
│
├── prompts/
│   └── LLM prompt templates
│
├── rag/
│   └── online retrieval and evidence processing modules
│
├── tests/
│   └── basic test files
│
├── utils/
│   └── helper utilities
│
├── deliverables/
│   ├── written_report/
│   ├── presentation/
│   └── video/
│
├── requirements.txt
├── run.sh
├── .env.example
├── .gitignore
└── README.md
```

---

# Installation and Setup

Follow these steps to install and run ClinIQ locally.

---

## 1. Prerequisites

Before running the project, make sure you have:

- Python 3.10 or higher
- Git
- pip
- A terminal or command prompt
- An API key for the LLM provider used in the project, such as Groq, OpenAI, or Gemini

Check Python:

```bash
python3 --version
```

Check Git:

```bash
git --version
```

If Python is not installed, install it first from the official Python website or through your operating system package manager.

---

## 2. Clone the Repository

```bash
git clone https://github.com/sunny-arch-82/clinIQ-agentic-clinical-pattern-recognition-decision-support-platform.git
cd clinIQ-agentic-clinical-pattern-recognition-decision-support-platform
```

---

## 3. Create a Virtual Environment

A virtual environment keeps the project dependencies separate from the rest of your system.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

After activation, your terminal should show something like:

```bash
(.venv)
```

---

## 4. Install Dependencies

Upgrade pip first:

```bash
pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

This installs the Python packages required for the Streamlit interface, agentic workflow, LLM access, online retrieval, data validation, and report generation.

---

## 5. Configure Environment Variables

The project includes a template file:

```text
.env.example
```

Create your own local `.env` file:

```bash
cp .env.example .env
```

Then open `.env` and add your API key values.

Example:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Only add the key for the provider you plan to use.

Important:

```text
Do not commit your real .env file to GitHub.
```

The `.env.example` file is only a template. The real `.env` file should stay local.

---

## 6. Run the Application

Start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

After running the command, Streamlit will show a local URL such as:

```text
http://localhost:8501
```

Open that URL in your browser.

---

## 7. Alternative Run Command

If the repository includes `run.sh`, you can also run:

```bash
bash run.sh
```

If permission is denied, run:

```bash
chmod +x run.sh
./run.sh
```

---

## 8. Run a Demo Case

Inside the Streamlit app:

1. Select one of the demo claim JSON cases.
2. Recommended demo case: `CASE-HIP-006`.
3. Click the button to run the analysis.
4. Watch the workflow progress through:
   - claim validation
   - pattern recognition
   - query planning
   - online evidence retrieval
   - evidence quality review
   - verification
   - decision intelligence
   - report generation
5. Open or download the generated PDF report.

---

## 9. Expected Output

After a successful run, ClinIQ generates a Decision Intelligence Report containing:

- Claim key details
- Patterns recognized
- Online evidence retrieved
- Internal verification requirements
- Evidence analysis summary
- Clinical evidence assessment
- Verification status
- Recommended next action
- Suggested reviewer actions

The system does not approve or deny claims. It produces decision-support output for a human reviewer.

---

## 10. Troubleshooting

### Issue: `streamlit: command not found`

Run:

```bash
pip install streamlit
```

Or reinstall all dependencies:

```bash
pip install -r requirements.txt
```

---

### Issue: API key error

Check that your `.env` file exists and contains the correct API key.

```bash
cat .env
```

Make sure the key name matches the one expected by the project.

---

### Issue: Module not found

Make sure your virtual environment is activated:

```bash
source .venv/bin/activate
```

Then reinstall dependencies:

```bash
pip install -r requirements.txt
```

---

### Issue: Port already in use

Run Streamlit on another port:

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

---

### Issue: PDF report not generated

Check the terminal logs for errors.

Common causes:

- Missing API key
- Internet connection issue during online retrieval
- Missing dependencies
- Invalid or incomplete demo case JSON

Reinstall dependencies if needed:

```bash
pip install -r requirements.txt
```

---

## Quick Start

For users who already have Python and Git installed:

```bash
git clone https://github.com/sunny-arch-82/clinIQ-agentic-clinical-pattern-recognition-decision-support-platform.git
cd clinIQ-agentic-clinical-pattern-recognition-decision-support-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app/streamlit_app.py
```

Then open the Streamlit URL shown in the terminal.

---

## Assessment Deliverables

The required assessment deliverables are included in the `deliverables/` folder.

```text
deliverables/
├── written_report/
│   ├── ClinIQ_Written_Report_APA.docx
│   └── ClinIQ_Written_Report_APA.pdf
│
├── presentation/
│   ├── ClinIQ_Presentation.pptx
│   └── ClinIQ_Presentation.pdf
│
└── video/
    ├── ClinIQ_Video_Presentation_Demo.mp4
    └── ClinIQ_Video_Presentation_Demo_link
```

