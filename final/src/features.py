"""Weekly-cutoff feature construction with leakage control."""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS = 40
KEYS = ["code_module", "code_presentation", "id_student"]
TARGET_COLUMNS = {"y_bin", "y4", "next_score", "final_result"}
CLASS_MAP = {"Pass": 0, "Withdrawn": 1, "Fail": 2, "Distinction": 3}
AT_RISK_CLASSES = {"Withdrawn", "Fail"}


def week_index(dates: pd.Series) -> pd.Series:
    return ((dates // 7) + 1).clip(0, WEEKS - 1).astype("int16")


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(KEYS) | TARGET_COLUMNS
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def _click_features(sv: pd.DataFrame, cutoff_week: int) -> pd.DataFrame:
    sv = sv[sv["week"] <= cutoff_week]
    if sv.empty:
        return pd.DataFrame(columns=KEYS)
    total = (
        sv.groupby(KEYS, as_index=False)["sum_click"].sum()
        .rename(columns={"sum_click": "clicks_total"})
    )
    weekly_sum = sv.groupby(KEYS + ["week"], as_index=False)["sum_click"].sum()
    active_weeks = (
        weekly_sum[weekly_sum["sum_click"] > 0]
        .groupby(KEYS, as_index=False)
        .size()
        .rename(columns={"size": "active_weeks"})
    )
    recent = (
        weekly_sum[weekly_sum["week"] >= cutoff_week - 1]
        .groupby(KEYS, as_index=False)["sum_click"].sum()
        .rename(columns={"sum_click": "clicks_recent"})
    )
    prior_end = cutoff_week - 2
    prior = (
        weekly_sum[weekly_sum["week"].between(1, prior_end)]
        .groupby(KEYS, as_index=False)["sum_click"].sum()
        .rename(columns={"sum_click": "clicks_prior"})
    )
    out = total.merge(active_weeks, on=KEYS, how="left")
    out = out.merge(recent, on=KEYS, how="left")
    out = out.merge(prior, on=KEYS, how="left")
    out["clicks_total"] = out["clicks_total"].fillna(0)
    out["active_weeks"] = out["active_weeks"].fillna(0)
    out["clicks_recent"] = out["clicks_recent"].fillna(0)
    out["clicks_prior"] = out["clicks_prior"].fillna(0)
    n_prior_weeks = max(1, prior_end)
    out["click_trend"] = out["clicks_recent"] / 2.0 - out["clicks_prior"] / n_prior_weeks
    out["mean_weekly_clicks"] = out["clicks_total"] / max(1, cutoff_week)
    return out.drop(columns=["clicks_recent", "clicks_prior"])


def _granular_features(sv: pd.DataFrame, vle: pd.DataFrame, cutoff_week: int) -> pd.DataFrame:
    sv = sv[sv["week"] <= cutoff_week]
    if sv.empty:
        return pd.DataFrame(columns=KEYS)
    vle_map = vle[["id_site", "activity_type"]].drop_duplicates("id_site")
    sv_a = sv.merge(vle_map, on="id_site", how="left")
    sv_a["activity_type"] = sv_a["activity_type"].fillna("other")
    gran = (
        sv_a.groupby(KEYS + ["activity_type"], as_index=False)["sum_click"].sum()
        .pivot(index=KEYS, columns="activity_type", values="sum_click")
        .reset_index()
    )
    gran.columns = [c if c in KEYS else f"act_{c}" for c in gran.columns]
    num_cols = [c for c in gran.columns if c not in KEYS]
    gran[num_cols] = gran[num_cols].fillna(0).astype("float64")
    return gran


def _assessment_features(
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    cutoff_week: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_day = cutoff_week * 7
    a = assessments[assessments["assessment_type"] != "Exam"].copy()
    sa = student_assessment[student_assessment["is_banked"] == 0].copy()
    sa = sa.merge(
        a[["id_assessment", "code_module", "code_presentation", "date", "weight"]],
        on="id_assessment",
        how="inner",
    )
    sa["week"] = week_index(sa["date_submitted"])

    past = sa[sa["week"] <= cutoff_week].sort_values(["date_submitted", "id_assessment"])
    agg = past.groupby(KEYS, as_index=False).agg(
        n_submissions=("id_assessment", "size"),
        score_mean=("score", "mean"),
        score_last=("score", "last"),
    )
    on_time = past[past["date"].notna()].copy()
    on_time["on_time"] = (on_time["date_submitted"] <= on_time["date"]).astype("int8")
    ratio = on_time.groupby(KEYS, as_index=False)["on_time"].mean()
    ratio = ratio.rename(columns={"on_time": "on_time_ratio"})
    agg = agg.merge(ratio, on=KEYS, how="left")

    due = a[a["date"].notna() & (a["date"] <= cutoff_day)]
    expected = due.groupby(["code_module", "code_presentation"], as_index=False).size()
    expected = expected.rename(columns={"size": "n_expected"})
    agg = agg.merge(expected, on=["code_module", "code_presentation"], how="left")
    agg["n_expected"] = agg["n_expected"].fillna(0)
    agg["n_missing"] = (agg["n_expected"] - agg["n_submissions"]).clip(lower=0)
    agg = agg.drop(columns=["n_expected"])

    future = (
        sa[sa["week"] > cutoff_week]
        .sort_values(["date_submitted", "id_assessment"])
        .groupby(KEYS, as_index=False)
        .first()[KEYS + ["score"]]
        .rename(columns={"score": "next_score"})
    )
    return agg, future


def _demographic_features(info: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    df = info.merge(
        reg[KEYS + ["date_registration", "date_unregistration"]],
        on=KEYS,
        how="left",
    )
    cats = ["highest_education", "imd_band", "age_band", "gender", "disability"]
    dummies = pd.get_dummies(df[cats], dummy_na=True, dtype="int8")
    df = pd.concat([df, dummies], axis=1)
    df["date_registration"] = df["date_registration"].fillna(0)
    return df


def build_features(tables: dict[str, pd.DataFrame], cutoff_week: int) -> pd.DataFrame:
    """One row per student-module active at cutoff_week, with targets and next_score."""
    if not 1 <= cutoff_week <= 39:
        raise ValueError(f"cutoff_week must be in 1..39, got {cutoff_week}")
    cutoff_day = cutoff_week * 7

    base = _demographic_features(tables["studentInfo"], tables["studentRegistration"])
    active = base["date_unregistration"].isna() | (base["date_unregistration"] >= cutoff_day)
    base = base[active].copy()
    base = base.drop(columns=["date_unregistration"])

    sv = tables["studentVle"].assign(week=lambda d: week_index(d["date"]))
    clicks = _click_features(sv, cutoff_week)
    granular = _granular_features(sv, tables["vle"], cutoff_week)
    assessment, future = _assessment_features(
        tables["assessments"], tables["studentAssessment"], cutoff_week
    )

    df = base.merge(clicks, on=KEYS, how="left")
    df = df.merge(granular, on=KEYS, how="left")
    df = df.merge(assessment, on=KEYS, how="left")
    df = df.merge(future, on=KEYS, how="left")

    fill_zero = [
        "clicks_total",
        "active_weeks",
        "click_trend",
        "mean_weekly_clicks",
        "n_submissions",
        "n_missing",
    ]
    for c in fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    for c in ["score_mean", "score_last", "on_time_ratio"]:
        if c in df.columns:
            df[c] = df[c].fillna(-1.0)
    act_cols = [c for c in df.columns if c.startswith("act_")]
    df[act_cols] = df[act_cols].fillna(0)

    df["y_bin"] = df["final_result"].isin(AT_RISK_CLASSES).astype("int8")
    df["y4"] = df["final_result"].map(CLASS_MAP).astype("int8")
    df["next_score"] = df["next_score"].astype("float64")

    df = df.drop(columns=["region"], errors="ignore")
    return df.reset_index(drop=True)
