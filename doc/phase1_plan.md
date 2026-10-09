# Phase 1 Implementation Plan: Ingestion, Relational Pipeline & Baseline Setup

## 1. Objectives & Scope

Phase 1 establishes the reproducible data engineering backbone of the AEW-DSS system:
- High-throughput ingestion of all raw OULAD relational tables using Polars LazyFrames.
- Strict mapping of arbitrary daily interaction offsets to 7-day sequential buckets.
- Handling pre-course engagement ($t=0$) and post-withdrawal temporal censoring.
- Weighted normalization of milestone assessment scores into weekly envelopes.
- Constructing the complete Cartesian product timeline grid ($N \times 40$).
- Fitting leakage-free demographic encoding pipelines solely on training partitions.
- Stratified 70% / 15% / 15% train/val/test cohort partitioning preserving presentation groupings.
- Serialization of immutable Parquet artifacts and baseline empirical priors $P_0(Y = k)$.

---

## 2. Engineering Roadmap & Sub-Tasks

```text
Sprint 1.1: Polars Ingestion & Schema Contracts
|-- Verify CSV integrity and cast strict dtypes
\-- Build lazy query graphs for VLE, assessments, and registrations

Sprint 1.2: Temporal Normalization Engine
|-- Collapse pre-course dates (date < 0) into Week 0
|-- Bucket in-term dates (date >= 0) into floor(date / 7) + 1
\-- Aggregate VLE volume: raw clicks, distinct sites, log1p transform

Sprint 1.3: Assessment Alignment & Score Weighting
|-- Inner join studentAssessment with assessments metadata
|-- Compute credit-weighted normalized score S_t in [0.0, 1.0]
\-- Handle missing scheduled exam dates using presentation duration

Sprint 1.4: Cartesian Grid Realization & Censoring Controls
|-- Expand cohort spine across full timeline grid [0 .. 39]
|-- Compute unregistration week t_withdraw
\-- Assert is_censored and freeze logs post-withdrawal

Sprint 1.5: Demographic Encoding, Partitioning & Contracts
|-- Fit OneHot, Ordinal, and StandardScaler on training split
|-- StratifiedGroupKFold cohort partitioning (70/15/15)
|-- Compute empirical cohort baseline priors P_0(Y = k)
\-- Run automated acceptance assertion suite
```

---

## 3. Formal Algorithmic Formulations

### 3.1 Week Mapping Logic
$$t = f(\text{date}) = \begin{cases} 0, & \text{if } \text{date} < 0 \\ \left\lfloor \frac{\text{date}}{7} \right\rfloor + 1, & \text{if } \text{date} \ge 0 \end{cases}$$

### 3.2 Weekly Assessment Normalization
$$S_{i,t} = \frac{\sum_{a \in A_{i,t}} w_a \cdot \frac{\text{score}_{a}}{100}}{\sum_{a \in A_{i,t}} w_a}$$
If an assessment has weight $w_a = 0$ (formative), it is assigned unit weight ($1.0$). If no submission occurs at week $t$, $S_{i,t} = \text{null}$.

### 3.3 Withdrawal Censoring
$$t_{\text{withdraw}} = \begin{cases} 0, & \text{if } \text{date\_unregistration} < 0 \\ \lfloor \text{date\_unregistration} / 7 \rfloor + 1, & \text{if } \text{date\_unregistration} \ge 0 \end{cases}$$
For all weeks $t > t_{\text{withdraw}}$, `is_censored = True`, `raw_clicks = 0`, and `normalized_score = null`.

---

## 4. Acceptance Criteria & Quality Gates

| Verification Gate | Validation Condition | Enforcement |
|---|---|---|
| **Timeline Completeness** | Total rows in dynamic grid equals $N_{\text{cohort}} \times 40$ | Hard pipeline halt |
| **Censoring Isolation** | Zero click records and scores for weeks $t > t_{\text{withdraw}}$ | Pipeline failure |
| **Score Normalization** | All non-null $S_{i,t} \in [0.0, 1.0]$ | Pipeline failure |
| **Partition Disjointness** | Student ID intersection between Train, Val, and Test is $\emptyset$ | Pipeline failure |
| **Pre-course Consolidation** | Minimum week is $0$; no negative week values present | Pipeline failure |
