"""
AEW-DSS Phase 1: High-Performance Data Loader & Ingestion Pipeline
Engine: Polars (Multi-threaded columnar execution) & Scikit-learn
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("aew_dss.data_loader")

TARGET_MAP: Dict[str, int] = {
    "Pass": 0,
    "Withdrawn": 1,
    "Fail": 2,
    "Distinction": 3,
}

RAW_SCHEMAS = {
    "studentInfo": {
        "code_module": pl.Categorical,
        "code_presentation": pl.Categorical,
        "id_student": pl.UInt32,
        "gender": pl.Categorical,
        "region": pl.Categorical,
        "highest_education": pl.Categorical,
        "imd_band": pl.Categorical,
        "age_band": pl.Categorical,
        "num_of_prev_attempts": pl.UInt8,
        "studied_credits": pl.UInt16,
        "disability": pl.Categorical,
        "final_result": pl.Categorical,
    },
    "studentRegistration": {
        "code_module": pl.Categorical,
        "code_presentation": pl.Categorical,
        "id_student": pl.UInt32,
        "date_registration": pl.Int32,
        "date_unregistration": pl.Int32,
    },
    "vle": {
        "id_site": pl.UInt32,
        "code_module": pl.Categorical,
        "code_presentation": pl.Categorical,
        "activity_type": pl.Categorical,
        "week_from": pl.Float32,
        "week_to": pl.Float32,
    },
    "studentVle": {
        "code_module": pl.Categorical,
        "code_presentation": pl.Categorical,
        "id_student": pl.UInt32,
        "id_site": pl.UInt32,
        "date": pl.Int32,
        "sum_click": pl.Int32,
    },
    "assessments": {
        "code_module": pl.Categorical,
        "code_presentation": pl.Categorical,
        "id_assessment": pl.UInt32,
        "assessment_type": pl.Categorical,
        "date": pl.Int32,
        "weight": pl.Float32,
    },
    "studentAssessment": {
        "id_assessment": pl.UInt32,
        "id_student": pl.UInt32,
        "date_submitted": pl.Int32,
        "is_banked": pl.UInt8,
        "score": pl.Float32,
    },
}


def map_date_to_week_expr(col_name: str = "date") -> pl.Expr:
    """
    Polars expression:
    t = 0 if date < 0 else floor(date / 7) + 1
    """
    return (
        pl.when(pl.col(col_name) < 0)
        .then(pl.lit(0, dtype=pl.Int16))
        .otherwise((pl.col(col_name) / 7).floor().cast(pl.Int16) + 1)
        .alias("week")
    )


def process_vle_interactions(
    vle_lazy: pl.LazyFrame,
    max_course_week: int = 39,
) -> pl.LazyFrame:
    """
    Aggregates VLE interaction streams into weekly activity metrics:
    C_t = sum(clicks), C_tilde = ln(1 + C_t).
    """
    logger.info("Building VLE weekly interaction aggregation query plan...")
    return (
        vle_lazy
        .with_columns(map_date_to_week_expr("date"))
        .filter(pl.col("week") <= max_course_week)
        .group_by(["code_module", "code_presentation", "id_student", "week"])
        .agg([
            pl.col("sum_click").sum().cast(pl.UInt32).alias("raw_clicks"),
            pl.col("id_site").n_unique().cast(pl.UInt16).alias("distinct_materials_visited"),
        ])
        .with_columns([
            (pl.col("raw_clicks").cast(pl.Float32) + 1.0).log().alias("log_clicks")
        ])
    )



def process_granular_vle_interactions(
    vle_lazy: pl.LazyFrame,
    vle_info_lazy: pl.LazyFrame,
    max_course_week: int = 39,
) -> pl.LazyFrame:
    return (
        vle_lazy
        .join(vle_info_lazy.select(["id_site", "activity_type"]), on="id_site", how="left")
        .with_columns(map_date_to_week_expr("date"))
        .filter(pl.col("week") <= max_course_week)
        .group_by(["code_module", "code_presentation", "id_student", "week", "activity_type"])
        .agg([
            pl.col("sum_click").sum().cast(pl.UInt32).alias("raw_clicks")
        ])
    )

def process_assessments(
    assessments_lazy: pl.LazyFrame,
    student_assessment_lazy: pl.LazyFrame,
    max_course_week: int = 39,
) -> pl.LazyFrame:
    """
    Aligns assessment submission dates to weekly buckets and computes
    credit-weighted normalized scores S_t in [0.0, 1.0].
    """
    logger.info("Building assessment alignment and score normalization query plan...")
    joined = (
        student_assessment_lazy
        .filter(pl.col("is_banked") == 0)
        .join(assessments_lazy, on="id_assessment", how="inner")
        .filter(pl.col("assessment_type") != "Exam")
        .with_columns(map_date_to_week_expr("date_submitted"))
        .filter(pl.col("week") <= max_course_week)
    )

    return (
        joined
        .with_columns([
            (pl.col("score") / 100.0).clip(0.0, 1.0).alias("score_norm"),
            pl.col("weight").fill_null(0.0).alias("clean_weight"),
        ])
        .with_columns([
            pl.when(pl.col("clean_weight") == 0.0)
            .then(1.0)
            .otherwise(pl.col("clean_weight"))
            .alias("effective_weight")
        ])
        .with_columns([
            (pl.col("score_norm") * pl.col("effective_weight")).alias("weighted_score_val")
        ])
        .group_by(["code_module", "code_presentation", "id_student", "week"])
        .agg([
            (pl.col("weighted_score_val").sum() / pl.col("effective_weight").sum())
            .cast(pl.Float32)
            .alias("normalized_score"),
            pl.col("id_assessment").count().cast(pl.UInt8).alias("assessments_submitted_count"),
        ])
    )


def build_dynamic_features_matrix(
    students_df: pl.DataFrame,
    weekly_vle_df: pl.DataFrame,
    weekly_scores_df: pl.DataFrame,
    max_weeks: int = 39,
) -> pl.DataFrame:
    """
    Expands student cohorts across weeks [0, max_weeks] in a complete Cartesian grid.
    Applies post-withdrawal zero-masking and censoring flags.
    """
    logger.info("Constructing Cartesian timeline grid and applying censoring controls...")
    base_students = students_df.select([
        "code_module", "code_presentation", "id_student",
        "date_unregistration", "final_result"
    ]).with_columns([
        pl.when(pl.col("date_unregistration").is_not_null())
        .then(
            pl.when(pl.col("date_unregistration") < 0)
            .then(0)
            .otherwise((pl.col("date_unregistration") / 7).floor() + 1)
        )
        .otherwise(None)
        .cast(pl.Int16)
        .alias("week_unregistered")
    ])

    weeks_grid = pl.DataFrame({"week": np.arange(0, max_weeks + 1, dtype=np.int16)})
    student_spine = base_students.join(weeks_grid, how="cross")

    grid_joined = (
        student_spine
        .join(
            weekly_vle_df,
            on=["code_module", "code_presentation", "id_student", "week"],
            how="left"
        )
        .join(
            weekly_scores_df,
            on=["code_module", "code_presentation", "id_student", "week"],
            how="left"
        )
    )

    return (
        grid_joined
        .with_columns([
            pl.when(
                pl.col("week_unregistered").is_not_null() &
                (pl.col("week") > pl.col("week_unregistered"))
            )
            .then(True)
            .otherwise(False)
            .alias("is_censored"),

            pl.col("raw_clicks").fill_null(0),
            pl.col("log_clicks").fill_null(0.0),
            pl.col("distinct_materials_visited").fill_null(0),
            pl.col("assessments_submitted_count").fill_null(0),
        ])
        .with_columns([
            pl.when(pl.col("is_censored"))
            .then(0)
            .otherwise(pl.col("raw_clicks"))
            .alias("raw_clicks"),

            pl.when(pl.col("is_censored"))
            .then(0.0)
            .otherwise(pl.col("log_clicks"))
            .alias("log_clicks"),

            pl.when(pl.col("is_censored"))
            .then(None)
            .otherwise(pl.col("normalized_score"))
            .alias("normalized_score"),
        ])
        .sort(["code_module", "code_presentation", "id_student", "week"])
    )


class ProductionDemographicTransformer(BaseEstimator, TransformerMixin):
    """
    Leakage-free demographic transformer fitted exclusively on training partitions.
    """
    def __init__(self):
        self.nominal_cols = ["gender", "region", "disability"]
        self.numeric_cols = ["studied_credits", "num_of_prev_attempts"]
        self.ordinal_education_order = [
            "No Formal quals",
            "Lower Than A Level",
            "A Level or Equivalent",
            "HE Qualification",
            "Post Graduate Qualification",
        ]
        self.ordinal_age_order = ["0-35", "35-55", "55<="]
        self.ordinal_imd_order = [
            "Missing", "0-10%", "10-20", "20-30%", "30-40%",
            "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
        ]

        self.edu_map = {v: i for i, v in enumerate(self.ordinal_education_order)}
        self.age_map = {v: i for i, v in enumerate(self.ordinal_age_order)}
        self.imd_map = {v: i for i, v in enumerate(self.ordinal_imd_order)}

        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
        self.scaler = StandardScaler(copy=True)

    def _ensure_pandas(self, X: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pl.DataFrame):
            return X.to_pandas()
        elif isinstance(X, pd.DataFrame):
            return X.copy()
        raise TypeError(f"Expected polars.DataFrame or pandas.DataFrame, got {type(X)}")

    def fit(self, df: pl.DataFrame | pd.DataFrame, y=None) -> ProductionDemographicTransformer:
        pdf = self._ensure_pandas(df)
        nominal_df = pdf[self.nominal_cols].astype(object).fillna("Missing")
        self.ohe.fit(nominal_df)

        numeric_df = pdf[self.numeric_cols].astype(np.float64).fillna(0.0)
        self.scaler.fit(numeric_df)
        return self

    def transform(self, df: pl.DataFrame | pd.DataFrame) -> np.ndarray:
        pdf = self._ensure_pandas(df)

        nominal_df = pdf[self.nominal_cols].astype(object).fillna("Missing")
        nom_enc = self.ohe.transform(nominal_df)

        edu_series = pdf["highest_education"].astype(object)
        edu_enc = edu_series.map(self.edu_map).fillna(-1).to_numpy(dtype=np.float32)[:, None]

        age_series = pdf["age_band"].astype(object)
        age_enc = age_series.map(self.age_map).fillna(-1).to_numpy(dtype=np.float32)[:, None]

        imd_series = pdf["imd_band"].astype(object).fillna("Missing")
        imd_series = imd_series.replace({"?": "Missing", "None": "Missing", "nan": "Missing"})
        imd_enc = imd_series.map(self.imd_map).fillna(self.imd_map["Missing"]).to_numpy(dtype=np.float32)[:, None]

        numeric_df = pdf[self.numeric_cols].astype(np.float64).fillna(0.0)
        num_enc = self.scaler.transform(numeric_df).astype(np.float32)

        return np.hstack([nom_enc, edu_enc, age_enc, imd_enc, num_enc])


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    empirical_priors: Dict[str, float]


def generate_stratified_cohort_splits(
    student_df: pl.DataFrame,
    seed: int = 42,
) -> SplitResult:
    """
    70/15/15 train/validation/holdout test partition stratified on outcome labels
    and grouped by course presentation cohorts to minimize contamination.
    """
    logger.info("Computing leakage-free stratified cohort partitions...")
    pdf = student_df.select([
        "code_module", "code_presentation", "id_student", "final_result"
    ]).to_pandas()

    pdf["presentation_group"] = (
        pdf["code_module"].astype(str) + "_" + pdf["code_presentation"].astype(str)
    )
    pdf["target"] = pdf["final_result"].map(TARGET_MAP)

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    groups = pdf["id_student"].to_numpy()
    X = pdf[["id_student"]].to_numpy()
    y = pdf["target"].to_numpy()

    train_indices, temp_indices = [], []
    for fold, (trn_idx, tst_idx) in enumerate(sgkf.split(X, y, groups)):
        if fold < 7:
            train_indices.extend(tst_idx)
        else:
            temp_indices.extend(tst_idx)

    train_idx = np.array(train_indices)
    temp_idx = np.array(temp_indices)

    sub_groups = groups[temp_idx]
    sub_X = X[temp_idx]
    sub_y = y[temp_idx]

    sub_sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_rel, test_rel = next(sub_sgkf.split(sub_X, sub_y, sub_groups))

    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    train_targets = y[train_idx]
    total_samples = len(train_targets)
    classes, counts = np.unique(train_targets, return_counts=True)

    inv_target_map = {v: k for k, v in TARGET_MAP.items()}
    priors = {inv_target_map[c]: float(cnt / total_samples) for c, cnt in zip(classes, counts)}

    logger.info(f"Empirical Baseline Priors P0(Y): {priors}")
    return SplitResult(train_idx, val_idx, test_idx, priors)


def verify_phase1_contract(
    dynamic_df: pl.DataFrame,
    demographics_df: pl.DataFrame,
    split_res: SplitResult,
) -> bool:
    """Executes contract assertions on Phase 1 output matrices."""
    logger.info("Executing Phase 1 contract assertion checks...")

    total_students = demographics_df.height
    expected_rows = total_students * 40
    assert dynamic_df.height == expected_rows, (
        f"Row count mismatch: expected {expected_rows}, got {dynamic_df.height}"
    )

    assert dynamic_df["week"].min() == 0, "Found weeks below 0"
    assert dynamic_df["week"].max() == 39, "Weeks exceed maximum bound of 39"

    valid_scores = dynamic_df["normalized_score"].drop_nulls()
    assert valid_scores.min() >= 0.0 and valid_scores.max() <= 1.0, (
        "Scores found outside normalized range [0.0, 1.0]"
    )

    censored_violations = dynamic_df.filter(
        pl.col("is_censored") & (pl.col("raw_clicks") > 0)
    ).height
    assert censored_violations == 0, (
        f"Leakage detected: {censored_violations} active interactions after withdrawal"
    )

    train_ids = set(demographics_df[split_res.train_idx]["id_student"])
    val_ids = set(demographics_df[split_res.val_idx]["id_student"])
    test_ids = set(demographics_df[split_res.test_idx]["id_student"])

    assert len(train_ids.intersection(val_ids)) == 0, "Train/Val ID contamination"
    assert len(train_ids.intersection(test_ids)) == 0, "Train/Test ID contamination"
    assert len(val_ids.intersection(test_ids)) == 0, "Val/Test ID contamination"

    logger.info("All Phase 1 contract assertions passed successfully.")
    return True


def run_pipeline(raw_data_dir: Path, output_dir: Path):
    """Orchestrates Phase 1 data pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning raw OULAD CSV files...")
    student_info = pl.scan_csv(raw_data_dir / "studentInfo.csv", schema=RAW_SCHEMAS["studentInfo"])
    student_reg = pl.scan_csv(raw_data_dir / "studentRegistration.csv", schema=RAW_SCHEMAS["studentRegistration"])
    student_vle = pl.scan_csv(raw_data_dir / "studentVle.csv", schema=RAW_SCHEMAS["studentVle"])
    vle_info = pl.scan_csv(raw_data_dir / "vle.csv", schema=RAW_SCHEMAS["vle"])
    assessments = pl.scan_csv(raw_data_dir / "assessments.csv", schema=RAW_SCHEMAS["assessments"])
    student_assessment = pl.scan_csv(raw_data_dir / "studentAssessment.csv", schema=RAW_SCHEMAS["studentAssessment"])

    base_students_df = (
        student_info
        .join(student_reg, on=["code_module", "code_presentation", "id_student"], how="left")
        .collect()
    )

    weekly_vle_df = process_vle_interactions(student_vle).collect()
    granular_vle_df = process_granular_vle_interactions(student_vle, vle_info).collect()
    weekly_scores_df = process_assessments(assessments, student_assessment).collect()

    dynamic_features_df = build_dynamic_features_matrix(
        base_students_df,
        weekly_vle_df,
        weekly_scores_df,
        max_weeks=39,
    )

    split_result = generate_stratified_cohort_splits(base_students_df)

    train_base_df = base_students_df[split_result.train_idx]
    demo_encoder = ProductionDemographicTransformer().fit(train_base_df)
    encoded_demographics = demo_encoder.transform(base_students_df)

    verify_phase1_contract(dynamic_features_df, base_students_df, split_result)

    logger.info("Persisting immutable Parquet and NumPy artifacts...")
    dynamic_features_df.write_parquet(
        output_dir / "dynamic_features_weekly.parquet",
        compression="zstd",
        statistics=True,
    )
    granular_vle_df.write_parquet(
        output_dir / "granular_vle.parquet",
        compression="zstd",
    )
    base_students_df.write_parquet(
        output_dir / "static_demographics_raw.parquet",
        compression="zstd",
    )
    np.save(output_dir / "encoded_demographics.npy", encoded_demographics)
    np.savez_compressed(
        output_dir / "cohort_splits.npz",
        train_idx=split_result.train_idx,
        val_idx=split_result.val_idx,
        test_idx=split_result.test_idx,
        priors=np.array([list(split_result.empirical_priors.items())]),
    )
    logger.info("Phase 1 pipeline completed successfully.")


if __name__ == "__main__":
    RAW_DIR = Path("./data/raw")
    OUT_DIR = Path("./data/processed")
    if (RAW_DIR / "studentInfo.csv").exists():
        run_pipeline(RAW_DIR, OUT_DIR)
    else:
        logger.info(f"Raw OULAD CSV files not found in {RAW_DIR}. Pipeline ready for execution once CSVs are present.")
