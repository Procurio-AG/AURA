# Future Experiments

Planned follow-ups to the completed experiment log (`../RESULTS.md`, EXP-001 through EXP-010). Not yet run. When executed, each gets logged in RESULTS.md as EXP-0xx with config, results, and artifacts, and the paper's future-scope section is updated accordingly.

Numbering continues from RESULTS.md. Costs are estimated OpenRouter spend at gpt-4o-mini pricing unless noted.

---

## EXP-011 - Report-quality rubric grading
**Priority:** high | **Cost:** ~$2-3 | **Effort:** low (evaluation script + LLM judge)

Grade governed vs ungoverned reports (n = 20 students) on a fixed rubric: evidence grounding (does every claim trace to a tool output?), uncertainty communication (are caveats present and substantive?), actionability (is the recommended action executable by a tutor?). LLM-as-judge with a strong model, plus spot-checked manual grading for judge validity.

**What it would show:** whether governance costs anything in report quality while fixing decisions - the missing human-factors half of the evaluation. Expected: governed reports are more conservative but better-grounded; ungoverned reports read confidently but cite less and hedge wrong.

## EXP-012 - Split-robustness replication
**Priority:** high | **Cost:** free | **Effort:** trivial (`--split group`)

Rerun the benchmark with a student-grouped random split instead of the 2014J presentation holdout.

**What it would show:** whether the benchmark table is an artifact of split choice. Expected: grouped split yields slightly higher AUC (presentation-level distribution shift removed); the NB < LR < LGBM ordering and the small LR-LGBM gap should hold either way.

## EXP-013 - Cross-LLM replication of the bias result
**Priority:** high | **Cost:** ~$5-15 (model-dependent) | **Effort:** low

Rerun the 100-student bias experiment with 2-3 additional models spanning tiers (e.g., gpt-4o or claude-sonnet class, plus an open-weights model via OpenRouter).

**What it would show:** whether the ungoverned bias is a small-model artifact or a structural property of LLM advisors. Atkinson's 43pp zero-shot result for GPT-4o suggests it persists at the frontier; quantifying the gap across tiers would materially strengthen the claim.

## EXP-014 - Post-hoc calibration of the ungoverned LLM
**Priority:** high | **Cost:** ~$3-5 | **Effort:** medium

Fit Platt scaling (logistic recalibration) to the ungoverned agent's self-reported probabilities on a held calibration set, then re-evaluate its decisions through the recalibrated estimates.

**What it would show:** whether statistical calibration can substitute for structural governance. If calibrated-LLM decisions approach oracle agreement, the governance architecture is overkill; if not (expected, because the LLM's ranking itself is noisy, and calibration cannot fix wrong orderings), structural governance wins outright. Either outcome is publishable.

## EXP-015 - Prompting ablation on the ungoverned agent
**Priority:** medium | **Cost:** ~$3 | **Effort:** low

Give the ungoverned agent progressively stronger prompts: base rates ("about 40% of students are at-risk"), conservative instructions, explicit threshold rules ("act only if clearly at risk").

**What it would show:** whether the bias is prompt-addressable. Expected: prompts shift the act rate but do not track the oracle across cost settings - killing the "just engineer the prompt" objection to structural governance.

## EXP-016 - Fairness audit of the governed policy
**Priority:** high | **Cost:** free | **Effort:** medium

Break down governed and ungoverned decisions, miss rates, and model probabilities by IMD deprivation band, disability, age band, and gender (the Bayer et al. line of work). Compute subgroup recall / false-positive differentials and, if disparate, whether cost-model tuning can equalize them.

**What it would show:** whether the 27% miss rate clusters in disadvantaged groups - a deployment-blocking question for any real institution.

## EXP-017 - Early-cutoff stress test (W04)
**Priority:** medium | **Cost:** ~$3 | **Effort:** low

Rerun the bias experiment at cutoff W04, where the model itself is weakest (AUC 0.70).

**What it would show:** whether the governed agent degrades gracefully with model uncertainty (it should - the policy inherits a weaker but still calibrated probability) and whether the ungoverned gap widens or narrows when less evidence is visible to the LLM.

## EXP-018 - Abstention behavior
**Priority:** medium | **Cost:** ~$3 + code | **Effort:** medium

Extend the policy with an explicit abstain action at low-information weeks (e.g., abstain when the probability sits inside an uncertainty band derived from calibration error). Evaluate how abstention trades coverage against the miss rate.

**What it would show:** whether "the advisor declines to judge" is the right early-week behavior - connecting to the calibration story and addressing PredAct-Bench's finding that hiding tool noise is the advisor's worst failure mode.

## EXP-019 - Bias evolution across cutoffs
**Priority:** low | **Cost:** ~$6-10 | **Effort:** low

Rerun the ungoverned half of the bias experiment at W12 and W39 (oracle side is free), reusing the caching design.

**What it would show:** whether the ungoverned LLM converges toward the model's decisions as the visible evidence grows (its estimates could anchor better on richer data) or stays fixed - i.e., whether the bias is an information problem or a judgement problem.

---

## Suggested order

EXP-012 (free robustness) -> EXP-014 (calibration vs governance, the sharpest scientific question) -> EXP-015 (prompting ablation) -> EXP-013 (cross-LLM) -> EXP-016 (fairness) -> EXP-011 (rubric) -> EXP-017/018/019.

EXP-012 + 014 + 015 together attack the three most likely reviewer objections for roughly $10 total.
