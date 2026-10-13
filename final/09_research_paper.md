# Calibrated Classical Models as Safety Governors for LLM Advisors: A Tool-Governed Early-Warning System for Student Attrition

**Author Name**
Department, University, City, Country
author@university.edu

*Working paper draft. Markdown in IEEE journal article structure; results and artifacts referenced from the companion experiment log (`RESULTS.md`).*

---

## Abstract

Early-warning systems for student attrition have historically relied on classical machine learning over learning-management-system data, while the emergence of large language model (LLM) agents offers an attractive conversational and investigative layer for the advisors who must act on such predictions. Recent evidence, however, shows that LLMs making autonomous risk judgements are systematically miscalibrated and prone to intervention bias. This paper presents a tool-governed agentic advisor over the Open University Learning Analytics Dataset (OULAD), in which an LLM agent handles retrieval and report writing while all risk judgements are structurally routed through a calibrated Logistic Regression model, with intervention decisions derived from an explicit cost model. A leakage-controlled benchmark at four cutoff weeks shows that the Logistic Regression primary model achieves an ROC-AUC of 0.77 at week 8 and 0.93 at week 39, within 0.04 of a gradient-boosted reference at every cutoff, while achieving the best calibration (expected calibration error 0.033 at the operating week). In an intervention-bias experiment on 100 held-out students against a cost-optimal oracle policy, the tool-governed agent achieves perfect agreement by construction, whereas an ungoverned LLM advisor with identical data access disagrees on 40% of decisions, missing 27% of students requiring intervention, over-intervening on 13%, and producing risk estimates with a mean absolute error of 0.29 relative to the calibrated model. A cost-sensitivity analysis further shows the ungoverned agent cannot adapt to institutional cost structures while the governed policy retunes exactly. The results indicate that classical probabilistic models remain the appropriate decision core for advisory agents, with the LLM contributing retrieval, explanation, and communication.

**Index Terms**— learning analytics, student attrition, early warning systems, large language models, intelligent agents, logistic regression, naive Bayes, calibration, human-AI interaction

---

## I. Introduction

Non-completion rates in distance higher education substantially exceed those of traditional institutions: in the Open University Learning Analytics Dataset (OULAD), over 52% of enrolled student-module instances end in withdrawal (31.2%) or failure (21.4%) [1]. Because disengagement in asynchronous learning is silent, institutions deploy early-warning systems (EWS) that flag at-risk students early enough for support interventions to matter; the upstream production system built on this data delivered weekly risk predictions to tutors for several years [2].

Machine-learning components for such systems are well studied, and the literature converges on a simple recipe: aggregate clickstream behavior into weekly cumulative features and apply linear or ensemble models, with binary at-risk formulations reaching ROC-AUC in the 0.75-0.90 band during weeks 1-10 [3], [4]. The deployment gap is elsewhere: predictions must reach a human advisor as answers, explanations, and recommended actions. Large language model (LLM) agents are a natural fit for this interface layer, since they can translate questions into database queries, assemble evidence, and draft intervention notes.

The problem is that LLMs asked to make the risk judgement themselves are demonstrably unsafe in this role. Atkinson [5] reports that zero-shot GPT-4o advisory agents on OULAD exhibit a 43-percentage-point intervention bias, recommending action in the majority of cases where a decision-theoretic oracle mandates inaction, and PredAct-Bench [6] shows that state-of-the-art LLMs paired with noisy risk tools fail to communicate tool uncertainty to instructors. These findings raise the central question of this work: if an LLM agent is unavoidable as the advisor interface, where should the boundary between LLM judgement and model judgement be drawn?

This paper answers that question empirically with a boundary drawn structurally rather than through prompting. We deploy an LLM agent with SQL query tools over the OULAD database and a strict rule: every risk verdict must come from a calibrated Logistic Regression model exposed as a tool, and every intervention decision is a deterministic function of that model's probability under an explicit two-parameter cost model. The LLM retrieves, explains, and writes; it cannot decide. We compare this *governed* advisor against an *ungoverned* variant (identical LLM and database access, self-judged risk) on the oracle-policy protocol of [5].

### Contributions

1. A leakage-controlled, multi-cutoff benchmark of a deliberately simple model ladder (Gaussian Naive Bayes, Logistic Regression, LightGBM) on a binary at-risk formulation of OULAD, quantifying the accuracy and calibration cost of an interpretable linear core (Section VI-A, VI-B).
2. A tool-governed agentic advisor architecture in which intervention decisions are structurally derived from calibrated model probabilities and a cost model, with a structured, evidence-cited report schema (Section IV-C).
3. An intervention-bias evaluation replicating the oracle-policy protocol on a tool-calling agent stack, showing 100% oracle agreement for the governed design versus 60% for the ungoverned LLM, with a cost-sensitivity analysis demonstrating that the ungoverned agent cannot adapt to institutional cost structures (Section VI-C, VI-D).

The work originates as an extension of a course project on Bayesian reasoning and logistic regression; all core predictive machinery remains Naive Bayes and Logistic Regression, which turns out to be a feature rather than a limitation for agent governance.

---

## II. Related Work

**OULAD benchmarks.** The dataset [1] comprises 32,593 student-module instances across 22 module-presentations with 10.6 million daily VLE click records. Published results cluster around ROC-AUC 0.75-0.90 for early-horizon binary dropout prediction [3], [4], with ensemble deep learning [7] and survival formulations [8] as extensions. A systematic review cataloguing OULAD benchmark studies is given in [9]. A recurring finding, confirmed by ablation in the survival benchmark of [8], is that temporal behavioral features dominate demographic features, which motivates our feature design.

**Leakage control.** The leakage-excluded evaluation protocol of LEAP [3] documents how assessment-related features inflate early-horizon performance when future information enters the feature matrix. Our pipeline inherits a three-rule leakage policy (Section III-B) distilled from a prior audit of an earlier iteration of this system, in which final-exam scores, in-sample calibration, and post-withdrawal evaluation inflated a 4-class macro F1 from 0.538 to 0.739.

**LLM advisors in education.** Susnjak [10] established the pattern of transparent ML predicting and ChatGPT generating prescriptive feedback. Atkinson [5] provides the direct motivation: on OULAD snapshots, zero-shot LLM advisory agents carry severe intervention bias that supervised calibration eliminates. PredAct-Bench [6] benchmarks LLM dialogue agents using noisy student-risk tools and finds they fail to convey tool noise to teachers. The present work sits between these poles: rather than calibrating the LLM, we remove the LLM from the decision path entirely and measure what that buys.

**Agents over databases.** Text-to-SQL agents are a mature pattern [11]; our contribution is not the agent machinery but the governance boundary and its evaluation.

---

## III. Dataset and Preprocessing

### A. Data

OULAD [1] consists of seven relational tables: student demographics and outcomes, module registrations with withdrawal dates, daily VLE clickstreams, VLE site metadata (activity type), assessment structure, and assessment scores. The prediction unit is the student-module instance. The primary target is binary: at-risk (final result Withdrawn or Fail) versus not at-risk (Pass or Distinction), with class balance 52.6% / 47.4% overall. The original 4-class formulation is retained only as a secondary reporting table.

### B. Leakage rules

Three rules are enforced at every cutoff week t:

1. Final-exam assessments (which nearly determine the outcome) are excluded from all features; only continuous coursework (TMA/CMA) is used.
2. Students who unregistered before week t are excluded from week-t evaluation pools (active-student masking).
3. The unregistration date never enters the feature matrix; for a student active at t it is future information. This rule was added after an audit caught the field leaking through the feature builder (logged as EXP-003 in the experiment log).

### C. Split

The test cohort is a temporal holdout: all 2014J module-presentations. Training uses all prior presentations. This evaluates generalization to an unseen future cohort, the deployment-relevant setting.

---

## IV. Methodology

### A. Feature construction

At each cutoff week t in {4, 8, 12, 39}, one flat feature vector is built per active student-module from data up to t: cumulative clicks, active weeks, mean weekly clicks, a recent-versus-prior click trend, per-activity-type cumulative clicks (forum, quiz, resource, and so on), assessment statistics (mean and last score, submission count, on-time ratio, missing-submission count against deadlines due by t), one-hot demographics (education, deprivation band, age band, gender, disability), prior attempts, studied credits, and registration date. This flat-aggregation design follows the consensus that weekly cumulative behavioral features carry the signal [3], [8]; it replaces a substantially more complex sequential Bayesian engine from the earlier iteration of this system, which contributed no measurable gain over its stacking layer.

### B. Model ladder

Three classifiers are trained on identical features: Gaussian Naive Bayes (the Bayesian baseline required by the project lineage), Logistic Regression with balanced class weights (the primary model), and LightGBM [12] (a gradient-boosted ceiling reference, explicitly out of scope as the deployed model). A ridge regressor predicts the next assessment score, evaluated with MAE, RMSE, and R-squared. The deployed Logistic Regression is deliberately not wrapped in post-hoc calibration because its logit link yields native probabilities; Section VI-B verifies this choice empirically.

### C. Tool-governed agent architecture

The advisor is a ReAct agent [13] over a read-only SQLite database containing the raw OULAD tables plus materialized weekly aggregates. Its toolset:

1. **SQL tools**: schema listing, schema inspection, and a sandboxed query tool that automatically caps results at 100 rows and prefers the materialized aggregates over the 10.6-million-row raw clickstream. (The row cap was introduced after an early experiment in which the ungoverned agent issued unbounded SELECT queries and exceeded the LLM context window on 37 of 100 cases; the fix reduced failures to 1 of 100, logged as EXP-008a/008b.)
2. **predict_risk**: returns the calibrated Logistic Regression probability with the top standardized-coefficient drivers for the queried student and week.
3. **predict_next_score**: the ridge regressor's next-score prediction.
4. **write_intervention_report**: assembles a structured report (Pydantic-validated) containing student identifier, cutoff week, model probability, risk band, evidence items each tagged with a source (sql or model), recommended action, rationale, and uncertainty caveats. The recommended action is computed inside the tool from the model probability and the cost model; the LLM supplies evidence, rationale, and caveats but not the verdict.

The **governed policy** maps probability to action levels derived from the cost model (Section IV-D): outreach at p >= c_i/(c_m + c_i), urgent outreach at the midpoint between that threshold and 1, monitoring at half the threshold, otherwise no action. With the default costs this yields outreach at 0.20, urgent at 0.60, monitor at 0.10.

The **ungoverned baseline** is the same LLM with the same SQL tools and no model tools; it estimates risk and chooses actions freely, producing the same report schema. This isolates the effect of the governance boundary while holding model, data access, and prompting constant.

### D. Oracle policy

Following [5], decisions are scored against a cost-optimal oracle. Let intervening cost c_i (normalized to 1) and missing an at-risk student cost c_m. Acting is optimal when the expected cost of inaction exceeds that of action:

p * c_m >= (1 - p) * c_i, i.e. p >= c_i / (c_m + c_i).

With c_m = 4 the oracle acts when p >= 0.20. The oracle is computed on the calibrated model probability; it is the decision-theoretic answer key given the model's information.

### E. Metrics

Discrimination: ROC-AUC, PR-AUC, F1, at-risk recall. Calibration: Brier score and 10-bin expected calibration error (ECE) on reliability diagrams. Agent evaluation: oracle agreement, over-intervention rate (act when the oracle does not), under-intervention rate (fail to act when the oracle does), action-level distribution, and mean absolute deviation of the ungoverned agent's self-estimated probability from the model probability.

---

## V. Experimental Setup

- **Data**: OULAD via figshare (CC BY 4.0), 7 tables, 10.6M click rows, materialized into SQLite with weekly aggregates (603k weekly-click rows; 2.4M weekly-activity rows).
- **Split**: presentation holdout (test = 2014J); train sizes 18,334 (W04) to 15,001 (W39); test sizes 9,219 to 7,521; test at-risk prevalence 0.408 (W04) to 0.274 (W39).
- **Models**: scikit-learn [14] GaussianNB and LogisticRegression (max_iter 3000, balanced weights, standardized features); LightGBM (300 trees, learning rate 0.05, balanced).
- **Agent**: LangChain 1.x / LangGraph ReAct agent; SQL tools over a read-only connection; gpt-4o-mini via OpenRouter at temperature 0 as the LLM for both advisor variants.
- **Bias experiment**: 100 students sampled from the held-out 2014J test cohort at cutoff W08 (seeded), both advisor variants, oracle at c_m in {2, 4, 8}, c_i = 1.
- All experiments, commands, and artifacts are logged in the companion experiment log (EXP-001 through EXP-010).

---

## VI. Results

### A. Discrimination benchmark

Table I reports the binary at-risk benchmark on the 2014J holdout.

**Table I. Binary at-risk performance on the 2014J presentation holdout.**

| Cutoff | NB AUC | LR AUC | LGBM AUC | LR F1 | LR at-risk recall | LR Brier |
|---|---|---|---|---|---|---|
| W04 | 0.678 | 0.696 | 0.736 | 0.569 | 0.554 | 0.220 |
| W08 | 0.720 | 0.768 | 0.783 | 0.606 | 0.558 | 0.187 |
| W12 | 0.752 | 0.801 | 0.808 | 0.638 | 0.632 | 0.176 |
| W39 | 0.878 | 0.927 | 0.934 | 0.760 | 0.779 | 0.097 |

Three observations. First, the Logistic Regression core lands within 0.007 to 0.040 AUC of LightGBM at every cutoff, so the accuracy sacrificed for an interpretable linear decision core is small. Second, the results sit inside the published early-horizon band for this dataset [3], [4], [7], confirming the pipeline is competitive without recurrent or boosting machinery in the deployed path. Third, the secondary 4-class multinomial Logistic Regression reaches macro F1 0.498 at W39, comparable to the 0.538 of the far more complex prior architecture on a similar protocol, reinforcing that the earlier system's sequential Bayesian engine added complexity without signal.

The ridge next-score regressor achieves MAE 12.6-13.8 points (R-squared near 0.1), reported as an honest lower bound: next-score point prediction from behavioral features alone is intrinsically hard, and the metric fulfills the regression-evaluation component of the project lineage.

### B. Calibration

Table II reports expected calibration error on the holdout.

**Table II. Expected calibration error (10 bins, lower is better).**

| Cutoff | Naive Bayes | Logistic Regression | LightGBM |
|---|---|---|---|
| W04 | 0.429 | 0.062 | 0.056 |
| W08 | 0.285 | **0.033** | 0.063 |
| W12 | 0.222 | 0.080 | 0.082 |
| W39 | 0.198 | 0.064 | 0.045 |

Gaussian Naive Bayes is severely miscalibrated (ECE 0.20-0.43): correlated weekly click features violate the conditional-independence assumption, and the accumulated posterior becomes overconfident. The Logistic Regression primary is the best-calibrated model at the agent's operating week (W08, ECE 0.033) and remains within 0.03-0.08 throughout. This is the quantitative basis for routing agent decisions through LR probabilities and for declining to deploy NB posteriors as decision inputs, despite NB's curricular role as the Bayesian baseline.

### C. Intervention-bias experiment

Table III reports the main result: 100 held-out students at W08, oracle with c_m = 4, c_i = 1.

**Table III. Advisor decisions versus the cost-optimal oracle (n = 100).**

| Variant | Act rate | Oracle agreement | Over-intervention | Under-intervention | Mean abs. p error | Errors |
|---|---|---|---|---|---|---|
| Oracle | 0.74 | - | - | - | - | - |
| Governed | 0.74 | **1.00** | 0.00 | 0.00 | 0 (structural) | 0 |
| Ungoverned | 0.60 | 0.60 | 0.13 | 0.27 | 0.289 | 1 |

The governed agent agrees with the oracle on every case by construction: its action is a pure function of the calibrated model probability under the same cost model, so no LLM behavior can perturb the verdict. The ungoverned agent disagrees on 40% of decisions in both directions: 27% of students who should receive intervention receive none (the error type the cost model prices at four times the intervention), and 13% receive unnecessary intervention. Its self-estimated probabilities deviate from the calibrated model by 0.29 on average, and its action distribution is severity-inflated: 52 of 99 decisions are urgent-outreach versus 21 under the governed policy. The single pipeline failure was a pathological provider response (546 tool calls in one message), not a governance artifact.

Notably, the bias direction differs from the pure over-intervention reported for zero-shot GPT-4o advisors in [5]: with SQL-tool access and a smaller model, miscalibration manifests in both directions plus severity inflation. LLM advisory bias should therefore be characterized as uncalibrated noise around the decision boundary, not a fixed directional tendency; in either form it is large and costly.

### D. Cost sensitivity

Table IV recomputes the oracle at three miss-to-intervention cost ratios (ungoverned actions cached and invariant).

**Table IV. Oracle act rate and ungoverned disagreement versus cost_miss (c_i = 1).**

| cost_miss | Act threshold | Oracle act rate | Ungoverned agreement | Over-int. | Under-int. |
|---|---|---|---|---|---|
| 2 | 0.333 | 0.46 | 0.62 | 0.26 | 0.12 |
| 4 | 0.200 | 0.74 | 0.60 | 0.13 | 0.27 |
| 8 | 0.111 | 0.95 | 0.59 | 0.03 | 0.38 |

The ungoverned agent's act rate is invariant at 0.60 across all cost regimes because it has no channel through which institutional costs enter its decision. The governed policy tracks the oracle exactly at every setting. The ungoverned error direction flips with the cost structure: when misses are expensive it under-intervenes on 38% of students needing outreach; when interventions are relatively expensive it over-intervenes on 26%. A raw LLM advisor is therefore wrong in whichever direction the institution's cost structure punishes most, whereas the governed design is a two-parameter policy instrument that retunes without retraining or re-prompting.

---

## VII. Discussion

### A. Findings

The results support a division of labor: classical probabilistic models for belief, LLM agents for labor. The linear core provides discrimination within 0.04 AUC of gradient boosting, the best calibration at the operating point, and coefficient-level explanations that a black box could not offer the advisor's report. The LLM provides query translation, evidence assembly, and narrative generation. The governance boundary converts the LLM from a decision maker into an interface, and the bias experiment quantifies what that conversion is worth: the difference between 40% decision error and zero.

### B. Limitations and threats to validity

1. **Single LLM and prompt**: both variants use gpt-4o-mini at temperature 0; stronger models or different prompts could narrow the ungoverned gap, though [5] reports severe bias for frontier zero-shot advisors as well.
2. **Oracle inherits the model**: the oracle is cost-optimal given the LR probability, not given the true outcome probability; the governed agent's 100% agreement is therefore structural, and its practical ceiling is the model's discrimination and calibration, not perfection.
3. **Simulated costs**: c_m and c_i are normalized abstractions; real intervention economics vary by institution and student.
4. **Temporal holdout only**: the presentation-based split was not cross-checked against student-grouped splits in the reported runs.
5. **No human factors**: report quality is schema-validated but not yet rubric-graded by human or LLM-as-judge evaluation, and no instructor-in-the-loop study was conducted.

---

## VIII. Conclusion and Future Scope

We presented an early-warning advisor in which an LLM agent's risk judgements are structurally delegated to a calibrated Logistic Regression, and showed that this boundary eliminates the intervention bias observed when the LLM judges freely, across all tested cost regimes. The classical core required by the project's curricular lineage (Naive Bayes and Logistic Regression) proved sufficient: within 0.04 AUC of gradient boosting, best-calibrated at the operating week, and interpretable enough to drive evidence-cited reports.

The immediate future scope is a prioritized experiment program, logged in full detail in the companion file `future_experiments.md`:

- **EXP-011**, report-quality rubric grading of governed versus ungoverned reports (evidence grounding, uncertainty communication, actionability), completing the human-factors half of the evaluation.
- **EXP-012**, split-robustness replication of the benchmark under student-grouped splits.
- **EXP-013**, cross-LLM replication of the intervention-bias result across model tiers, testing whether the ungoverned gap is a small-model artifact or a structural property of LLM advisors.
- **EXP-014**, post-hoc calibration (Platt scaling) of the ungoverned agent's self-reported probabilities, testing whether statistical calibration can substitute for structural governance.
- **EXP-015**, a prompting ablation on the ungoverned agent, testing whether prompt engineering can close the gap that governance closes.
- **EXP-016**, a fairness audit of the governed policy across deprivation bands, disability, and age subgroups, extending [15].
- **EXP-017**, an early-cutoff stress test at W04 examining whether the governed agent degrades gracefully when model discrimination is weakest.
- **EXP-018**, abstention behavior, permitting the advisor to decline judgement at low-information weeks.
- **EXP-019**, bias evolution across W04/W08/W12/W39 cutoffs as the LLM's visible evidence grows.

Together these target the three principal objections to the present study: single-model dependence (EXP-013), the possibility of cheaper remedies than structural governance (EXP-014, EXP-015), and unexamined subgroup effects (EXP-016).

---

## References

[1] J. Kuzilek, M. Hlosta, and Z. Zdrahal, "Open University Learning Analytics dataset," *Scientific Data*, vol. 4, art. 170171, 2017.

[2] C. Herodotou et al., "The engagement of university teachers with predictive learning analytics," *Computers & Education*, vol. 174, 2021.

[3] LEAP: "When can we trust early warnings? Leakage-excluded early outcome prediction from LMS interaction logs," arXiv:2605.25794.

[4] N. Tertulino and C. Almeida, "Evaluating federated learning for at-risk student prediction," arXiv:2508.18316, 2025.

[5] T. Atkinson, "Deterministic decisions for high-stakes AI: A zero-egress pipeline with the deployability of RAG and the accuracy of ML," arXiv:2606.29280, 2026.

[6] Y. AlRabah, D. Hakkani-Tur, et al., "PredAct-Bench: Benchmarking LLM agents paired with noisy student-risk prediction tools," arXiv:2608.02372, 2026.

[7] J. Jha, I. Ghergulescu, and G. Moldovan, "OULAD MOOC dropout and result prediction using ensemble deep learning and regression," in *Proc. CSEDU*, 2019, pp. 154-164.

[8] da Silva, J. Eicher, and C. Longo, "A unified survival benchmark for temporal dropout risk prediction," arXiv:2604.08870, 2026.

[9] N. Alhakbani and M. Alnassar, "Open learning analytics: A systematic review of benchmark studies using OULAD," in *Proc. ICMLT*, 2022.

[10] T. Susnjak, "A prescriptive learning analytics framework: Beyond predictive modelling and onto explainable AI with prescriptive analytics and ChatGPT," *Int. J. Artificial Intelligence in Education*, 2023.

[11] LangChain, "Build a SQL agent," LangChain documentation. [Online]. Available: https://docs.langchain.com/oss/python/langchain/sql-agent

[12] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems*, 2017.

[13] S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," arXiv:2210.03629, 2022.

[14] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[15] S. Bayer, M. Hlosta, and M. Fernandez, "Do existing algorithms serve everyone equally? Using dynamic fair machine learning to improve the identification of at-risk students," in *Proc. AIED*, 2021.
