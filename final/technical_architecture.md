# Technical Architecture - Detailed Design and Rationale

Companion documents: `problem_statement.md` (what and why), `09_research_paper.md` (results), `future_experiments.md` (next steps). Code lives in `src/`.

---

## 1. System Overview

```
┌────────────────────────────────────────────────────────────────────┐
│ LAYER 3 - AGENT (LangGraph ReAct, LLM = gpt-4o-mini)               │
│                                                                    │
│   sql tools (read-only, row-capped) ──┐                            │
│   predict_risk (calibrated LR)  ──────┼──► ReAct loop ──► answer   │
│   predict_next_score (ridge)    ──────┤         │                  │
│   write_intervention_report ─────────┘         ▼                  │
│                                    InterventionReport (Pydantic)   │
│                                    action = policy(p, costs)      │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ predict_proba / predict
┌──────────────────────────▼─────────────────────────────────────────┐
│ LAYER 2 - ML CORE                                                  │
│   features_w{t}: flat weekly-cumulative vectors, one per student   │
│   ladder: GaussianNB (baseline) / LR (deployed) / LightGBM (ref)   │
│   ridge regressor for next-score                                   │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ reads / materializes
┌──────────────────────────▼─────────────────────────────────────────┐
│ LAYER 1 - DATA                                                     │
│   OULAD CSVs ──► oulad.db (SQLite, read-only for the agent)        │
│   raw tables + weekly_clicks + weekly_activity aggregates           │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Design Principles (and why)

### 2.1 Flat weekly aggregation over sequential modeling

**Decision.** One flat feature vector per student per cutoff week (cumulative clicks, per-activity-type clicks, assessment stats, demographics). No RNNs, no sequential likelihood accumulation.

**Reason.** The prior architecture (AURA, archived) used a Dual-Path Gaussian Naive Bayes engine with temporal discounting and milestone weight calibration. Its standalone Macro F1 was 0.2994 versus 0.5384 for the stack on top of it - the sequential engine contributed complexity without signal. The literature independently converged on weekly aggregation + simple models (survival benchmark, LEAP, federated LR; see paper Section II). Aggregation also sidesteps the autocorrelated-clicks problem that forced the old gamma discounting hack: summing correlated features into a logistic model is statistically unremarkable, while summing correlated log-likelihoods is not.

### 2.2 Calibration is the product, accuracy is a constraint

**Decision.** The deployed model is the one with the best-calibrated probabilities at the operating week, not the highest AUC.

**Reason.** The entire agent governance mechanism consumes `predict_proba` output. A model that ranks well but outputs overconfident probabilities would shift every policy threshold. Empirically LR had the best ECE at W08 (0.033), beating LightGBM; NB was catastrophically miscalibrated (0.285 at W08), which is itself a textbook illustration of the independence assumption breaking on correlated features.

### 2.3 Structural governance over prompt governance

**Decision.** The LLM cannot set the intervention action. The action is computed inside the `write_intervention_report` tool from the model probability and the cost model. The LLM writes evidence, rationale, and caveats.

**Reason.** Prompts are suggestions; code paths are guarantees. The bias experiment (paper Table III) shows what happens when the boundary is only nominal: the ungoverned variant deviates on 40% of decisions. The governed design's 100% oracle agreement is not an empirical accident but a structural property - there is no code path through which the LLM alters the verdict.

### 2.4 The cost model is the single source of decision truth

**Decision.** Both the oracle (evaluation answer key) and the governed policy thresholds derive from the same two parameters: `cost_miss` and `cost_intervention`. Action levels: outreach at p >= t, urgent at (1+t)/2, monitor at t/2, where t = cost_intervention / (cost_miss + cost_intervention).

**Reason.** An early consistency check (EXP-005) caught the original hand-picked thresholds (outreach at 0.40) disagreeing with the cost-derived oracle (act at 0.20) - the governed agent was under-intervening by construction. Unifying both on the cost model makes the deployed policy and the evaluation oracle definitionally consistent, and turns the institution's economics into two tunable parameters rather than retraining or re-prompting.

### 2.5 Leakage discipline inherited, not reinvented

**Decision.** Three enforced rules (Section 4.2 below); evaluation on a temporal presentation holdout.

**Reason.** The prior AURA audit documented three leaks that inflated 4-class Macro F1 from 0.538 to 0.739. One new leak (`date_unregistration` in the feature matrix) was caught during this build by GaussianNB refusing NaN input - a nice example of why the honest pipeline should fail loudly rather than impute silently.

---

## 3. Layer 1 - Data

### 3.1 Ingestion (`src/data_loader.py`)

- Downloads the OULAD zip from figshare (CC BY 4.0), extracts 7 CSVs, type-casts all columns explicitly.
- **Reason for explicit dtypes**: silent type coercion on 10.6M-row files is a classic source of downstream bugs; explicit schemas fail fast.

### 3.2 SQLite materialization

- `oulad.db` stores the 7 raw tables plus `weekly_clicks` (603k rows) and `weekly_activity` (2.4M rows) aggregates.
- **Reason**: the agent must never scan the 10.6M-row `studentVle` in a conversation loop. The weekly tables compress the clickstream ~17x and answer the questions advisors actually ask ("how many clicks in week 6?"). `evaluate.py` later adds `features_w{t}` and `split_w{t}` tables so the agent can also query model inputs.
- **Read-only enforcement**: the agent connects through `sqlite3.connect("file:...?mode=ro", uri=True)`. Prompt-level "do not modify" instructions are not enforcement.
- **Threading**: LangGraph executes tools in worker threads. Plain SQLite connections raised `ProgrammingError` on close from foreign threads; the SQL engine therefore uses `check_same_thread=False` plus `NullPool` (fresh connection per query, no cross-thread pooling). This was diagnosed from connection-close tracebacks during the agent smoke tests (EXP-006).

---

## 4. Layer 2 - ML Core

### 4.1 Feature construction (`src/features.py`)

One row per student-module active at cutoff week t. Feature groups:

| Group | Columns | Source tables |
|---|---|---|
| Click volume | clicks_total, active_weeks, mean_weekly_clicks, click_trend (recent 2-week mean minus prior-window mean) | studentVle |
| Granular activity | act_forumng, act_quiz, act_oucontent, ... per activity type | studentVle + vle |
| Assessment | score_mean, score_last, n_submissions, on_time_ratio, n_missing (deadlines due by t minus submissions) | studentAssessment + assessments |
| Demographics | one-hot highest_education, imd_band (with NaN indicator), age_band, gender, disability; num_of_prev_attempts, studied_credits | studentInfo |
| Registration | date_registration | studentRegistration |

**Reason for `n_missing`**: deadlines due but not submitted is one of the strongest early behavioral signals, and it is available before any score exists.
**Reason for sentinel -1 on score features** (no submissions yet): distinguishes "no evidence" from "score of zero", which matters at early cutoffs. `n_submissions` lets the model learn the difference.

### 4.2 Leakage rules (enforced)

1. Exam assessments excluded everywhere - they nearly determine the target.
2. Active-student masking - students who unregistered before t are excluded from the week-t pool; predicting "withdrawn" for someone who already left is trivially right and meaningless.
3. `date_unregistration` excluded from features - future information for active students. (Caught in this build, EXP-003.)

### 4.3 Model ladder (`src/models.py`)

| Model | Role | Why it is / is not deployed |
|---|---|---|
| GaussianNB | Baseline | Curricular mandate (Bayes theorem, Naive Bayes). Kept as the honest floor and as a calibration cautionary tale: ECE 0.20-0.43. Not deployed. |
| LogisticRegression (balanced, standardized) | **Deployed primary** | Best calibration at the operating week (ECE 0.033 at W08); within 0.007-0.040 AUC of LightGBM; standardized coefficients give per-student driver attributions, which the agent's report tool surfaces as evidence. |
| LightGBM | Ceiling reference | Quantifies the cost of staying linear. Not deployed: slightly worse calibration at W08 and no coefficient-level explanations for reports. |
| Ridge | Next-score regressor | Covers the regression-metrics requirement of the original course topics (MAE, RMSE, R2). |

**Reason for standardized features**: puts LR coefficients on a common scale so `top_drivers` attributions (contribution = coefficient x z-score) are comparable across features.

### 4.4 Split

Temporal presentation holdout: test = all `2014J` presentations, train = everything earlier.
**Reason**: deployment means "predict next year's cohort from this year's data". A random split would leak presentation-level effects (module difficulty drift, cohort composition) in both directions.

---

## 5. Layer 3 - Agent

### 5.1 Tool inventory (`src/agent/tools.py`)

| Tool | Backing | Notes |
|---|---|---|
| `sql_db_list_tables` | SQLDatabase | schema discovery |
| `sql_db_schema` | SQLDatabase | schema inspection |
| `sql_query` | custom, row-capped | see 5.2 |
| `predict_risk(student_id, week)` | calibrated LR bundle | returns p, band, top drivers |
| `predict_next_score(student_id)` | ridge bundle | 0-100 prediction |
| `write_intervention_report(...)` | policy + Pydantic | final structured output |

`ModelBackend` loads the joblib bundle and the `features_w{t}` table into memory; model tools are therefore single-digit-millisecond calls, so the LLM never waits on training or large I/O.

### 5.2 The row-capped SQL tool

`sql_query` auto-wraps any query lacking a LIMIT in `SELECT * FROM (q) LIMIT 100`, fetches at most 101 rows, and returns compact JSON with a `truncated` flag.

**Reason**: EXP-008a. The ungoverned agent issued `SELECT * FROM studentVle` (10.6M rows) and similar unbounded queries; results flooded the message history past the 8 MB request limit and killed 37 of 100 runs. SQLite's subquery flattening pushes the LIMIT down into the scan, so the wrapper costs nothing. The tool description also nudges the LLM toward the aggregate tables. Failures dropped to 1 of 100 (a pathological provider response, not a query problem).

### 5.3 Report schema (`src/agent/report.py`)

`InterventionReport` fields: student_id, cutoff_week, p_at_risk, risk_band, evidence (list of items each tagged `sql` or `model`), recommended_action, action_rationale, uncertainty_caveats.

**Reasons**:
- Evidence items must carry a source tag - the structural defense against the LLM inventing facts; anything not from a tool has nowhere to hide.
- `uncertainty_caveats` is a required field, because PredAct-Bench's central finding is that advisors fail by hiding tool noise from instructors.
- Pydantic validation means malformed reports are impossible to emit; downstream evaluation scripts consume them programmatically.

### 5.4 Advisor variants

- `GovernedAdvisor` (`src/agent/advisor.py`): full toolset; system prompt forbids self-judged risk; `report_for_student` is the deterministic path used by the bias evaluation (model probability, policy action, LLM-written rationale and caveats).
- `UngovernedAdvisor` (`src/agent/advisor_baseline.py`): identical LLM, temperature, database access, and report schema; only the model tools and the governance prompt are removed. The LLM estimates p and chooses the action itself.
**Reason for the symmetric design**: the comparison isolates exactly one variable - who makes the decision. Any measured difference is attributable to governance, not to capability or context differences.

### 5.5 LLM configuration

OpenRouter endpoint, `gpt-4o-mini`, temperature 0, 120s timeout, 2 retries, recursion limit 30 on the ungoverned agent's ReAct loop.
**Reason**: temperature 0 for decision reproducibility; the recursion limit bounds worst-case cost per student (learned from a pathological 546-tool-call response, EXP-008b).

---

## 6. Evaluation Harness

### 6.1 Benchmark (`src/evaluate.py`)

Loops cutoffs, builds features, trains the ladder, computes AUC / PR-AUC / F1 / recall / Brier, plus the 4-class secondary table and the ridge regressor metrics. Persists `features_w{t}` + `split_w{t}` into SQLite and the agent bundle (`models/agent_bundle_w{t}.joblib`) fitted on train data only.

### 6.2 Calibration (`src/calibration.py`)

10-bin reliability diagrams and ECE per model per cutoff. Motivated by Section 2.2: the deployed-model choice needed a quantitative basis.

### 6.3 Bias evaluation (`src/bias_eval.py`)

Oracle from the cost model; governed actions computed from the same model; ungoverned actions from the LLM. Two operational design decisions, both learned the hard way:

- **Incremental, resumable per-student caching** (`ungoverned_detail_w{t}.csv`, append per student): a 100-student LLM run takes tens of minutes and can be interrupted; caching makes reruns and cost-sensitivity sweeps (EXP-009) free.
- **Cost parameters plumbed into both oracle and policy recomputation**, so sensitivity analyses recompute consistently instead of drifting from the deployed defaults.

---

## 7. File Map

```
src/
  data_loader.py     download, typed loading, SQLite build (raw + weekly aggregates)
  features.py        leakage-controlled weekly-cutoff feature construction
  models.py          model ladder, calibration wrapper, coefficient-based drivers
  evaluate.py        benchmark harness; agent bundle training; DB materialization
  calibration.py     reliability diagrams + ECE
  bias_eval.py       oracle policy, governed/ungoverned comparison, caching
  agent/
    tools.py         ModelBackend, LangChain tools, row-capped SQL toolkit, LLM factory
    report.py        InterventionReport / EvidenceItem schemas
    advisor.py      governed advisor (chat + deterministic per-student reports)
    advisor_baseline.py  ungoverned advisor (same everything, no governance)
```

## 8. Reproduction

```powershell
python -m venv .venv
& .venv\Scripts\pip.exe install -r requirements.txt
& .venv\Scripts\python.exe -m src.data_loader --download     # OULAD -> oulad.db
& .venv\Scripts\python.exe -m src.evaluate                   # benchmark + agent bundle
& .venv\Scripts\python.exe -m src.calibration                # ECE + reliability diagrams
$env:OPENROUTER_API_KEY = "<key>"
& .venv\Scripts\python.exe -m src.bias_eval --n 100 --cutoff 8   # bias experiment
& .venv\Scripts\python.exe -m src.agent.advisor "<question>"    # chat with the advisor
```

## 9. Implementation Lessons (each tied to an incident)

1. **Fail loudly on leakage**: the NaN crash that exposed `date_unregistration` did what silent imputation would have hidden.
2. **Bound tool outputs**: an agent with unbounded SQL access will eventually SELECT * a 10M-row table; cap at the tool layer, not in the prompt.
3. **Derive thresholds from the objective**: hand-picked action thresholds will disagree with the evaluation oracle; derive both from one cost model.
4. **Make long LLM runs resumable**: append-only caching turned a fragile 100-call experiment into a free sensitivity analysis.
5. **Enforce read-only at the connection, not the prompt**: `mode=ro` SQLite URIs are one line of code and absolute.
