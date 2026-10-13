"""Reliability diagrams and expected calibration error for the model ladder."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.data_loader import DATA_DIR, load_tables
from src.evaluate import make_split
from src.features import build_features, feature_columns
from src.models import build_classifier_pipelines, classifier_metrics

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_CUTOFFS = [4, 8, 12, 39]


def expected_calibration_error(y, p, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == 0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p > lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / len(p)) * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def run(cutoffs=DEFAULT_CUTOFFS, data_dir: Path = DATA_DIR,
        out_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    tables = load_tables(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fig, axes = plt.subplots(
        1, len(cutoffs), figsize=(4 * len(cutoffs), 4.2), sharey=True,
    )
    for ax, c in zip(np.atleast_1d(axes), cutoffs):
        df = build_features(tables, c)
        feats = feature_columns(df)
        is_test = make_split(df, "presentation")
        train, test = df[~is_test], df[is_test]
        X_te, y_te = test[feats], test["y_bin"].to_numpy()
        for name, pipe in build_classifier_pipelines().items():
            pipe.fit(train[feats], train["y_bin"])
            p = pipe.predict_proba(X_te)[:, 1]
            rows.append({
                "cutoff_week": c,
                "model": name,
                "ece": expected_calibration_error(y_te, p),
                **classifier_metrics(y_te, p),
            })
            frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="uniform")
            ax.plot(mean_pred, frac_pos, marker="o", label=name)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Week {c:02d}")
        ax.set_xlabel("predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("observed at-risk rate")
    fig.suptitle("Reliability diagrams (test = 2014J presentation holdout)")
    fig.tight_layout()
    fig_path = out_dir / "calibration.png"
    fig.savefig(fig_path, dpi=150)
    ece_table = pd.DataFrame(rows)
    csv_path = out_dir / "calibration_ece.csv"
    ece_table.to_csv(csv_path, index=False)
    print(ece_table.round(4).to_string(index=False))
    print(f"figure: {fig_path}")
    print(f"table: {csv_path}")
    return ece_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration analysis")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    args = parser.parse_args()
    run(cutoffs=args.cutoffs)


if __name__ == "__main__":
    main()
