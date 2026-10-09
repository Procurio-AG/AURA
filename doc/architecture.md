# System Architecture & Design Specification

## 1. Overview & Core Philosophy

The Adaptive Early Warning & Decision Support System (AEW-DSS) addresses academic analytics challenges on the Open University Learning Analytics Dataset (OULAD) by decoupling high-frequency continuous engagement modeling from low-frequency milestone assessment modeling and static demographic features.

```text
+-----------------------------------------------------------------------------------------+
|                                    Raw OULAD Records                                    |
| [studentInfo]   [studentRegistration]   [studentVle]   [assessments]   [studentAssess]  |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                    Phase 1: Ingestion & Relational Transformation                       |
| - Polars Multi-Threaded Lazy Execution                                                  |
| - Pre-Course Week 0 Normalization (date < 0 -> t=0, date >= 0 -> floor(date/7) + 1)     |
| - Dynamic Weekly Cartesian Grid Expansion (Cohorts x Weeks [0..T_max])                   |
| - Withdrawal Freezing & Temporal Right-Censoring (is_censored = True for t > t_unreg)    |
| - Stratified Group Partitioning (70% Train, 15% Validation, 15% Holdout Test)           |
+-----------------------------------------------------------------------------------------+
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
+-----------------------------------------+   +-------------------------------------------+
| Dynamic Sequence Features (Parquet)     |   | Static Demographic Matrices (Parquet/NPY) |
| - Weekly log-clicks: ln(1 + C_t)        |   | - One-Hot Encodings (gender, region, etc.)|
| - Weighted assessment scores: S_t       |   | - Ordinal Encodings (education, imd, age) |
| - Censoring indicators & material count |   | - Scaled credits & attempt counts         |
+-----------------------------------------+   +-------------------------------------------+
                       │                                           │
                       ▼                                           │
+-----------------------------------------+                        │
| Phase 2 & 3: Dual-Path Naive Bayes      |                        │
| - Path A (No assessment): P(C_t | Y=k)  |                        │
| - Path B (Assessment): P(C_t, S_t | Y=k)|                        │
| - Weight array W_t exponential update   |                        │
| - Prior-to-posterior propagation        |                        │
+-----------------------------------------+                        │
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
| Phase 4: Meta-Multinomial Stacking Classifier                                           |
| - Feature concatenation: x_i^(t) = [ Demographics || P_t(Pass, Withdrawn, Fail, Dist) ]|
| - Cost-sensitive multinomial logistic loss: omega_k = N / (4 * N_k)                     |
| - Calibrated 4-class outcome probabilities                                              |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
| Phase 5 & 6: Application Layer & Decision Support Engine                                |
| [Early Warning Alerts]  [Dropout Forecasting]  [Triage Queue]  [Counterfactual What-If] |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Ingestion & Transformation Subsystem

### 2.1 Composite Identity Resolution
Because students frequently re-enroll across modules and presentations, single student identifiers (`id_student`) cannot serve as standalone keys. All relational merges enforce the composite primary key:
$$\mathcal{K} = (\text{code\_module}, \text{code\_presentation}, \text{id\_student})$$

### 2.2 Temporal Bucketing
Let $\text{date} \in \mathbb{Z}$ represent the relative day offset from course launch:
$$t = f(\text{date}) = \begin{cases} 0, & \text{if } \text{date} < 0 \\ \lfloor \text{date} / 7 \rfloor + 1, & \text{if } \text{date} \ge 0 \end{cases}$$
- Week $0$ aggregates all pre-course orientation interactions.
- Weeks $1 \dots T_{\max}$ ($T_{\max} \approx 39$) map consecutive 7-day operational windows.

### 2.3 Censoring & Withdrawal Freeze
When a student unregisters at week $t_{\text{withdraw}}$:
- All interaction values for weeks $t > t_{\text{withdraw}}$ are masked to $0$ and flagged with `is_censored = True`.
- In the Bayesian inference engine, posterior beliefs are frozen forward:
  $$P_t(Y = k) = P_{t_{\text{withdraw}}}(Y = k) \quad \forall t > t_{\text{withdraw}}$$
  This prevents artificial zero-click bias from corrupting post-withdrawal likelihoods.

---

## 3. Core Engine Formulation

### 3.1 Dual-Path Naive Bayes Likelihoods
- **Path A (Behavior-Only Weeks)**:
  $$P(E_t \mid Y = k) = \frac{1}{\sqrt{2\pi\sigma_{k,t}^2}} \exp\left(-\frac{(\tilde{C}_t - \mu_{k,t})^2}{2\sigma_{k,t}^2}\right)$$
- **Path B (Assessment Milestone Weeks)**:
  $$P(E_t \mid Y = k) = P(\tilde{C}_t \mid Y = k) \cdot \frac{1}{\sqrt{2\pi\tau_{k,t}^2}} \exp\left(-\frac{(S_t - \nu_{k,t})^2}{2\tau_{k,t}^2}\right)$$

### 3.2 Exponential Milestone Weighting
$$P_t(Y = k) = \frac{P_{t-1}(Y = k) \cdot [P(E_t \mid Y = k)]^{w_t}}{\sum_{j=1}^4 \left( P_{t-1}(Y = j) \cdot [P(E_t \mid Y = j)]^{w_t} \right)}$$
Where $w_t \in [0.4, 0.6]$ for orientation weeks, $w_t = 1.0$ for regular instruction, and $w_t \in [1.8, 2.5]$ for assessment submission deadlines.

---

## 4. Meta-Multinomial Stacking Layer

At any evaluation milestone $t$, the dynamic state vector is concatenated with static demographic descriptors:
$$\mathbf{x}_i^{(t)} = \left[ \mathbf{d}_i \,\Vert{}\, P_t(Y = \text{Pass}) \,\Vert{}\, P_t(Y = \text{Withdrawn}) \,\Vert{}\, P_t(Y = \text{Fail}) \,\Vert{}\, P_t(Y = \text{Distinction}) \right]$$

Multinomial logistic regression with cost-sensitive weighting balances rare distinctions and fail classes:
$$P(Y = k \mid \mathbf{x}_i^{(t)}) = \frac{\exp(\boldsymbol{\beta}_k^T \mathbf{x}_i^{(t)} + \alpha_k)}{\sum_{j=1}^4 \exp(\boldsymbol{\beta}_j^T \mathbf{x}_i^{(t)} + \alpha_j)}$$
