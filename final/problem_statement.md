# Problem Statement, Solution, and What Makes It Stand Out

---

## 1. The Problem

Distance-education students drop out at high rates and they drop out silently. In the Open University Learning Analytics Dataset (OULAD), 31.2% of student-module instances end in withdrawal and another 21.4% in failure, while the learning itself happens asynchronously: no classroom, no face-to-face warning signs. A student who stops engaging in week 5 is often only discovered when they fail the first tutor-marked assessment or never return at all.

An early-warning system (EWS) is supposed to close this gap: flag at-risk students early enough that a support intervention can still change the outcome. Two things make this hard in practice:

1. **The prediction problem.** From noisy behavioral signals (daily clickstreams, sparse assessment submissions, demographics), estimate the probability a student will withdraw or fail - early, when intervention still matters, not in week 39 when it is too late.
2. **The delivery problem.** A probability does not help anyone by itself. A tutor or support advisor needs to ask questions of the data, see the evidence, and receive a justified, actionable recommendation. This interface layer is increasingly built with LLM agents - and the research shows LLMs making risk judgements themselves are systematically miscalibrated and intervention-biased.

## 2. Constraints

The project descends from a course project with a fixed mandate: the core methods must be Bayes Theorem, Naive Bayes, Logistic Regression, and regression metrics. The prior iteration (AURA, archived in `../archive/`) met this with a Dual-Path Gaussian Naive Bayes engine plus stacking; after a rigorous leakage audit its honest 4-class Macro F1 was 0.538, with the sequential Bayesian engine contributing almost nothing on its own (0.299). The mandate stays; the complexity does not.

## 3. The Final Problem Statement

> Build an early-warning advisor for student attrition on OULAD in which:
>
> 1. at each cutoff week t in {4, 8, 12, 39}, and for every student still active at t, the system estimates P(at-risk), where at-risk = final outcome Withdrawn or Fail;
> 2. the predictive core is Naive Bayes and Logistic Regression (plus a LightGBM reference and a ridge regressor covering regression metrics);
> 3. an LLM agent serves as the advisor interface - querying the database, assembling evidence, writing intervention reports - but every risk verdict is structurally routed through the calibrated Logistic Regression, and every intervention decision is derived from an explicit cost model;
> 4. the safety of this arrangement is measured: does tool-governance eliminate the intervention bias that free-judging LLM advisors exhibit?

## 4. How We Solved It

### 4.1 The diagnosis first

The archived AURA results contained their own lessons. The dual-path Bayesian engine added complexity without signal (0.299 alone vs 0.538 stacked). The 4-class target conflated two different phenomena - withdrawal is behavioral disengagement, failure is academic struggle - and crushed recall on the overlapping classes (Distinction 0.22, Fail F1 0.42). The fix was subtraction: delete the engine, the temporal discounting, the milestone calibration; switch to binary at-risk (the literature's standard, where published AUC sits at 0.75-0.90 for early weeks); keep only what earned its place - the leakage discipline, the weekly feature engineering, the granular activity joins.

### 4.2 Three layers

1. **Data layer.** OULAD in SQLite: raw tables plus materialized weekly aggregates, so agent queries are fast and the 10.6M-row clickstream never enters a conversation.
2. **ML core.** Flat weekly-cumulative features per cutoff week; GaussianNB baseline, Logistic Regression primary (deployed), LightGBM ceiling reference, ridge next-score regressor. Presentation holdout (2014J) as the test cohort.
3. **Agent layer.** A ReAct agent with row-capped read-only SQL tools, `predict_risk` / `predict_next_score` model tools, and a `write_intervention_report` tool that computes the action from the model probability under the cost model. The LLM retrieves, explains, writes. It cannot decide.

### 4.3 The results

**Prediction (binary at-risk, 2014J holdout):**

| Cutoff | NB AUC | LR AUC | LightGBM AUC |
|---|---|---|---|
| W04 | 0.678 | 0.696 | 0.736 |
| W08 | 0.720 | 0.768 | 0.783 |
| W12 | 0.752 | 0.801 | 0.808 |
| W39 | 0.878 | 0.927 | 0.934 |

**Calibration (ECE, lower is better):** LR is the best-calibrated model at the operating week W08 (0.033); Naive Bayes is severely miscalibrated (0.20-0.43) - the quantitative reason the agent consumes LR probabilities.

**Governance (100 held-out students vs a cost-optimal oracle):**

| | Oracle agreement | Missed students | Unnecessary interventions |
|---|---|---|---|
| Governed agent | **100%** | 0% | 0% |
| Ungoverned LLM | 60% | 27% | 13% |

The ungoverned agent's risk estimates deviate from the calibrated model by 0.29 on average, it inflates urgency 5x, and its act rate is frozen at 0.60 regardless of the institution's cost structure - while the governed policy retunes exactly through two parameters.

Full experiment configs, artifacts, and the two failure-and-fix cycles are logged in `../RESULTS.md` (EXP-001 through EXP-010).

## 5. How It Stands Out

1. **First of its kind on this dataset.** No published work deploys a tool-governed LangChain-style agent over OULAD; the adjacent literature (Atkinson; PredAct-Bench) documents the failure modes of ungoverned LLM advisors but not this remedy. We built the remedy and measured it.
2. **Governance is structural, not rhetorical.** Most "LLM + ML" projects put the model behind a prompt ("please use the tool"). Here the action is computed inside the tool from the model probability; there is no code path for the LLM to override it. The 100% oracle agreement is a property of the architecture, not a lucky run.
3. **The course-mandated methods are load-bearing, not decorative.** Naive Bayes is the calibration cautionary tale; Logistic Regression is the deployed brain - best-calibrated at the operating week, within 0.04 AUC of gradient boosting, and its coefficients power the evidence citations in every report. The "old" methods turn out to be exactly what makes the "new" AI layer safe.
4. **A cost-tunable policy instrument.** Changing the institutional economics (how bad a missed student is versus an unnecessary outreach) retunes the entire advisor's behavior through two parameters - something a free-judging LLM demonstrably cannot do.
5. **Honesty as methodology.** Leakage rules enforced and documented (one new leak caught and fixed mid-build), temporal holdout, the ungoverned baseline given identical data access and prompting so the comparison isolates one variable, and limitations stated in the paper rather than buried.
6. **A negative result reported plainly.** The bias direction we measured (mixed under- and over-intervention plus severity inflation) differs from the pure over-intervention in the literature - a nuance that matters for anyone deploying these systems and that only shows up when you actually run the experiment.
