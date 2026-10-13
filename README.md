# OULAD Tool-Governed Early Warning System (EWS)

An intelligent early warning and advisory system for student attrition on the **Open University Learning Analytics Dataset (OULAD)**.

The project features a **3-Layer Architecture** combining calibrated machine learning with a **tool-governed LangGraph ReAct agent**, eliminating the intervention bias inherent in unconstrained LLM advisors.

---

## 1. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: AGENT ADVISOR (LangGraph ReAct, LLM = gpt-4o-mini)            │
│   • Read-only SQLite queries over OULAD                                │
│   • predict_risk: Calibrated Logistic Regression tool                  │
│   • predict_next_score: Ridge regression tool                          │
│   • write_intervention_report: Pydantic structured output             │
│   • Action policy: Governed by cost matrix C_action vs C_miss          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ predict_proba / predict
┌───────────────────────────────────▼────────────────────────────────────┐
│ LAYER 2: ML PREDICTIVE CORE                                            │
│   • Binary At-Risk formulation: P(Withdrawn or Fail)                   │
│   • Cutoffs: Weeks t in {4, 8, 12, 39}                                 │
│   • Model Ladder: GaussianNB (baseline) -> Calibrated LR (deployed)    │
│                   -> LightGBM (reference)                              │
│   • Rigorous Leakage Elimination (no final exams, temporal masks)      │
│   • Calibration: Platt scaling & Isotonic regression (ECE, Brier)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ SQL queries
┌───────────────────────────────────▼────────────────────────────────────┐
│ LAYER 1: DATA & STORAGE                                                │
│   • Raw OULAD CSVs in data/raw/ -> SQLite oulad.db                     │
│   • Precomputed weekly_clicks and weekly_activity aggregates           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Layout

- `final/`: **Core Research & Engine (Frozen / Read-Only)**
  - `final/problem_statement.md`: Problem statement, leakage diagnosis, and design rationale.
  - `final/technical_architecture.md`: In-depth 3-layer architectural specification.
  - `final/09_research_paper.md`: Full empirical research paper and experimental results.
  - `final/future_experiments.md`: Research backlog for subsequent validation sprints.
  - `final/agent_bundle_w08.joblib`: Trained Week 08 model bundle (LR, GNB, Ridge, preprocessors).
  - `final/src/`: Core engine implementations (data loader, features, models, calibration, agent).
- `src/`: Root adapter package exposing `final/src` functionality with root path defaults.
- `legacy/`: Preserved legacy multi-class AURA experimental code (for academic auditability).
- `scripts/`: Operational utilities (e.g., `scripts/ingest_data.py`).
- `data/`: Raw CSVs (`data/raw/`) and processed artifacts.
- `notebooks/`: Interactive analysis and demonstration notebooks.

---

## 3. Quickstart

### Installation
```bash
pip install -r requirements.txt
```

### Data Ingestion & Database Setup
Download the OULAD dataset from Figshare and materialize the SQLite database:
```bash
python scripts/ingest_data.py --download
```
Or if you already have CSVs in `data/raw/`:
```bash
python scripts/ingest_data.py
```

### Running the Governed Advisor
To run the LangGraph advisor with an OpenRouter/OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
python -m src.agent.advisor --student 12345 --module BBB --presentation 2014J --cutoff 8
```

---

## 4. Key Findings

1. **Governance Eliminates LLM Intervention Bias**:
   - An ungoverned LLM advisor exhibits severe intervention bias (~60% agreement with decision oracle, over-intervening due to risk aversion).
   - The tool-governed advisor achieves **100% agreement** with the decision-theoretic oracle across varying cost regimes.
2. **Predictive Accuracy**:
   - Calibrated Logistic Regression achieves ROC-AUC **0.781** at Week 08 and **0.870** at Week 39, trailing complex LightGBM by less than 0.015 AUC while offering full interpretability.
