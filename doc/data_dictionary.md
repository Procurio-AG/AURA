# AEW-DSS Data Dictionary & Feature Catalog

## 1. Raw OULAD Entity Schemas

### 1.1 `studentInfo.csv`
Primary source of demographic attributes and final outcome labels.

| Column | Type | Description | Values / Range |
|---|---|---|---|
| `code_module` | String (Cat) | Module identifier code | 'AAA' through 'GGG' |
| `code_presentation` | String (Cat) | Presentation identifier (Year + Term) | '2013J', '2014B', '2014J', etc. |
| `id_student` | UInt32 | Unique student identifier | Integer |
| `gender` | String (Cat) | Gender of student | 'M', 'F' |
| `region` | String (Cat) | Geographic region | Geographic territories |
| `highest_education` | String (Cat) | Pre-entry qualification level | 'No Formal quals' to 'Post Graduate' |
| `imd_band` | String (Cat) | Index of Multiple Deprivation band | '0-10%', '10-20%', ..., '90-100%' |
| `age_band` | String (Cat) | Student age bracket | '0-35', '35-55', '55<=' |
| `num_of_prev_attempts` | UInt8 | Number of prior module attempts | $\ge 0$ |
| `studied_credits` | UInt16 | Total enrolled credit load | e.g., 60, 120, 240 |
| `disability` | String (Cat) | Declared disability status | 'N', 'Y' |
| `final_result` | String (Cat) | Academic outcome label | 'Pass', 'Withdrawn', 'Fail', 'Distinction' |

### 1.2 `studentRegistration.csv`
Contains enrollment and formal unregistration day offsets.

| Column | Type | Description |
|---|---|---|
| `code_module` | String (Cat) | Module identifier |
| `code_presentation` | String (Cat) | Presentation identifier |
| `id_student` | UInt32 | Student identifier |
| `date_registration` | Int32 | Days relative to start (negative = registered early) |
| `date_unregistration` | Int32 (Nullable) | Days relative to start when student withdrew |

### 1.3 `studentVle.csv`
Daily interaction click-stream records.

| Column | Type | Description |
|---|---|---|
| `code_module` | String (Cat) | Module identifier |
| `code_presentation` | String (Cat) | Presentation identifier |
| `id_student` | UInt32 | Student identifier |
| `id_site` | UInt32 | VLE resource identifier |
| `date` | Int32 | Day relative to presentation start |
| `sum_click` | Int32 | Total clicks recorded on this resource |

### 1.4 `assessments.csv`
Assessment calendar and weighting metadata.

| Column | Type | Description |
|---|---|---|
| `code_module` | String (Cat) | Module identifier |
| `code_presentation` | String (Cat) | Presentation identifier |
| `id_assessment` | UInt32 | Unique assessment identifier |
| `assessment_type` | String (Cat) | 'TMA' (Tutor Marked), 'CMA' (Computer Marked), 'Exam' |
| `date` | Int32 (Nullable) | Planned submission date |
| `weight` | Float32 | Percentage weight in module grade (0 to 100) |

### 1.5 `studentAssessment.csv`
Student submission timestamps and recorded marks.

| Column | Type | Description |
|---|---|---|
| `id_assessment` | UInt32 | Assessment identifier |
| `id_student` | UInt32 | Student identifier |
| `date_submitted` | Int32 | Day relative to module start of submission |
| `is_banked` | UInt8 | Flag indicating assessment was carried over |
| `score` | Float32 (Nullable)| Recorded score (0 to 100) |

---

## 2. Dynamic Weekly Feature Catalog (`dynamic_features_weekly.parquet`)

Grid granularity: One record per `(code_module, code_presentation, id_student, week)` where `week` $\in [0, 39]$.

| Feature Name | Dtype | Formula / Definition | Bounds |
|---|---|---|---|
| `week` | Int16 | Sequential week ($0 = \text{pre-course}, 1..39 = \text{term}$) | $[0, 39]$ |
| `raw_clicks` | UInt32 | $C_{i,t} = \sum \text{sum\_click}$ in week $t$ | $[0, \infty)$ |
| `log_clicks` | Float32 | $\tilde{C}_{i,t} = \ln(1 + C_{i,t})$ | $[0.0, \infty)$ |
| `distinct_materials_visited` | UInt16 | Unique count of `id_site` accessed in week $t$ | $[0, \infty)$ |
| `normalized_score` | Float32 (Nullable) | $S_{i,t} = \frac{\sum w_a (\text{score}_a / 100)}{\sum w_a}$ | $[0.0, 1.0]$ |
| `assessments_submitted_count` | UInt8 | Submissions completed within week $t$ | $[0, 255]$ |
| `is_censored` | Boolean | True if $t > t_{\text{withdraw}}$ | True / False |

---

## 3. Target Variable Specification

| Class Name | Integer Index | Prevalence | Description |
|---|---|---|---|
| **Pass** | `0` | ~38% | Completed all module requirements with passing grade |
| **Withdrawn** | `1` | ~31% | Unregistered from the course before or during the term |
| **Fail** | `2` | ~21% | Completed the module but did not meet passing criteria |
| **Distinction** | `3` | ~10% | Surpassed high-performance evaluation thresholds |
