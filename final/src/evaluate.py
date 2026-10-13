"""Benchmark harness: NB vs LR vs LightGBM at multiple cutoffs, plus regressor and agent bundles."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import DATA_DIR, DB_PATH, load_tables
from src.features import KEYS, build_features, feature_columns
from src.models import (
    build_classifier_pipelines,
    build_regressor,
    classifier_metrics,
    regressor_metrics,
    save_bundle,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
TEST_PRESENTATION = "2014J"
DEFAULT_CUTOFFS = [4, 8, 12, 39]
AGENT_CUTOFF = 8


def make_split(df: pd.DataFrame, split: str) -> pd.Series:
    if split == "presentation":
        return df["code_presentation"] == TEST_PRESENTATION
    if split == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df["id_student"]))
        mask = pd.Series(False, index=df.index)
        mask.iloc[test_idx] = True
        return mask
    raise ValueError(f"unknown split: {split}")


def _write_to_sqlite(df: pd.DataFrame, table: str, db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        df.to_sql(table, con, index=False, if_exists="replace")
    finally:
        con.close()


def run_benchmark(
    data_dir: Path = DATA_DIR,
    db_path: Path = DB_PATH,
    out_dir: Path = OUTPUTS_DIR,
    models_dir: Path = MODELS_DIR,
    cutoffs=DEFAULT_CUTOFFS,
    split: str = "presentation",
    agent_cutoff: int = AGENT_CUTOFF,
) -> pd.DataFrame:
    tables = load_tables(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    agent_bundle = None

    for c in cutoffs:
        df = build_features(tables, c)
        feats = feature_columns(df)
        is_test = make_split(df, split)
        train, test = df[~is_test], df[is_test]
        print(
            f"\n=== cutoff W{c:02d}: {len(train):,} train / {len(test):,} test "
            f"(test at-risk rate {test['y_bin'].mean():.3f}) ==="
        )

        X_tr, X_te = train[feats], test[feats]
        y_tr, y_te = train["y_bin"].to_numpy(), test["y_bin"].to_numpy()

        for name, pipe in build_classifier_pipelines().items():
            pipe.fit(X_tr, y_tr)
            p = pipe.predict_proba(X_te)[:, 1]
            m = classifier_metrics(y_te, p)
            rows.append({"cutoff_week": c, "model": name, **m})
            print(f"  {name:<20} AUC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}  "
                  f"F1={m['f1']:.4f}  recall={m['at_risk_recall']:.4f}  brier={m['brier']:.4f}")

        multi = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        )
        multi.fit(X_tr, train["y4"])
        macro_f1 = f1_score(
            test["y4"], multi.predict(X_te), average="macro", zero_division=0
        )
        rows.append({"cutoff_week": c, "model": "logistic_regression_4class", "roc_auc": np.nan,
                     "pr_auc": np.nan, "f1": macro_f1, "at_risk_recall": np.nan, "brier": np.nan})
        print(f"  {'4-class LR (report)':<20} macro F1={macro_f1:.4f}")

        reg_mask_tr, reg_mask_te = train["next_score"].notna(), test["next_score"].notna()
        if reg_mask_tr.sum() > 0 and reg_mask_te.sum() > 0:
            reg = build_regressor()
            reg.fit(
                X_tr[reg_mask_tr], train.loc[reg_mask_tr, "next_score"]
            )
            m = regressor_metrics(
                test.loc[reg_mask_te, "next_score"],
                reg.predict(X_te[reg_mask_te]),
            )
            rows.append({"cutoff_week": c, "model": "ridge_next_score",
                         "roc_auc": np.nan, "pr_auc": np.nan, "f1": np.nan,
                         "at_risk_recall": np.nan,
                         "brier": m["rmse"],
                         "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]})
            print(f"  {'ridge next-score':<20} MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.3f} "
                  f"(n_test={int(reg_mask_te.sum())})")

        split_df = df[KEYS + ["y_bin"]].copy()
        split_df["is_test"] = is_test.to_numpy()
        _write_to_sqlite(df, f"features_w{c:02d}", db_path)
        _write_to_sqlite(split_df, f"split_w{c:02d}", db_path)

        if c == agent_cutoff:
            primary = build_classifier_pipelines()["logistic_regression"]
            primary.fit(X_tr, y_tr)
            reg_full = build_regressor()
            reg_mask = train["next_score"].notna()
            reg_full.fit(X_tr[reg_mask], train.loc[reg_mask, "next_score"])
            agent_bundle = {
                "classifier": primary,
                "regressor": reg_full,
                "feature_cols": feats,
                "cutoff": c,
                "split": split,
            }

    benchmark = pd.DataFrame(rows)
    bench_path = out_dir / "benchmark.csv"
    benchmark.to_csv(bench_path, index=False)
    print(f"\nbenchmark table saved to {bench_path}")

    if agent_bundle is not None:
        save_bundle(agent_bundle, models_dir / f"agent_bundle_w{agent_cutoff:02d}.joblib")
        print(f"agent bundle saved to {models_dir / f'agent_bundle_w{agent_cutoff:02d}.joblib'}")
    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark harness")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument("--split", choices=["presentation", "group"], default="presentation")
    parser.add_argument("--agent-cutoff", type=int, default=AGENT_CUTOFF)
    args = parser.parse_args()
    run_benchmark(
        data_dir=args.data_dir,
        db_path=args.db,
        out_dir=args.out_dir,
        cutoffs=args.cutoffs,
        split=args.split,
        agent_cutoff=args.agent_cutoff,
    )


if __name__ == "__main__":
    main()
