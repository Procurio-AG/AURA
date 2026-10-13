"""Intervention-bias evaluation: oracle policy vs governed vs ungoverned agent actions."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.tools import (
    DB_PATH,
    DEFAULT_CUTOFF,
    MODELS_DIR,
    ModelBackend,
    action_from_p,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"

ACTION_LEVELS = {
    "none": 0,
    "monitor": 1,
    "outreach": 2,
    "urgent_outreach": 3,
}


def oracle_act(p: np.ndarray, cost_miss: float, cost_intervention: float) -> np.ndarray:
    """Act iff E[cost | act] < E[cost | don't act]."""
    threshold = cost_intervention / (cost_miss + cost_intervention)
    return p >= threshold


def _load_test(backend: ModelBackend, db_path: Path):
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        features = pd.read_sql_query(
            f"SELECT * FROM features_w{backend.cutoff:02d}", con
        )
        split = pd.read_sql_query(
            f"SELECT * FROM split_w{backend.cutoff:02d}", con
        )
    test_keys = split[split["is_test"] == 1][
        ["code_module", "code_presentation", "id_student"]
    ]
    return features.merge(test_keys, on=["code_module", "code_presentation", "id_student"])


def run(n: int = 60, cutoff: int = DEFAULT_CUTOFF, seed: int = 42,
        cost_miss: float = 4.0, cost_intervention: float = 1.0,
        run_llm: bool = True, db_path: Path = DB_PATH, models_dir: Path = MODELS_DIR,
        out_dir: Path = OUTPUTS_DIR) -> dict:
    backend = ModelBackend(cutoff, db_path, models_dir)
    test = _load_test(backend, db_path)
    sample = test.sample(n=min(n, len(test)), random_state=seed).reset_index(drop=True)
    X = sample[backend.feature_cols]
    p = backend.bundle["classifier"].predict_proba(X)[:, 1]
    oracle = oracle_act(p, cost_miss, cost_intervention)
    governed = np.array([action_from_p(float(pi), cost_miss, cost_intervention) for pi in p])

    results = {
        "config": {
            "n": int(len(sample)), "cutoff": cutoff, "seed": seed,
            "cost_miss": cost_miss, "cost_intervention": cost_intervention,
        },
        "oracle_act_rate": float(oracle.mean()),
        "governed": _summarize(oracle, governed, None),
    }
    rows = pd.DataFrame({
        "code_module": sample["code_module"],
        "code_presentation": sample["code_presentation"],
        "id_student": sample["id_student"],
        "model_p_at_risk": p,
        "oracle_act": oracle.astype(int),
        "governed_action": governed,
    })

    if run_llm:
        from src.agent.advisor_baseline import UngovernedAdvisor

        ung = UngovernedAdvisor(db_path, models_dir, cutoff)
        detail_path = out_dir / f"ungoverned_detail_w{cutoff:02d}.csv"
        done: set[int] = set()
        if detail_path.exists():
            existing = pd.read_csv(detail_path)
            done = set(existing["id_student"].astype(int))
        else:
            pd.DataFrame(
                columns=["id_student", "ungoverned_action", "ungoverned_p_at_risk"]
            ).to_csv(detail_path, index=False)
        for i, sid in enumerate(rows["id_student"]):
            sid = int(sid)
            if sid in done:
                print(f"[{i + 1}/{len(rows)}] {sid}: cached, skipping", flush=True)
                continue
            action, est = "error", float("nan")
            try:
                reports = ung.report_for_student(sid, cutoff)
                r = reports[0]
                action, est = r["recommended_action"], float(r["p_at_risk"])
            except Exception as exc:
                print(f"[{i + 1}/{len(rows)}] {sid}: FAILED {exc}", flush=True)
            else:
                print(
                    f"[{i + 1}/{len(rows)}] {sid}: {action} (p_est={est:.2f})",
                    flush=True,
                )
            pd.DataFrame(
                [{"id_student": sid, "ungoverned_action": action,
                  "ungoverned_p_at_risk": est}]
            ).to_csv(detail_path, mode="a", header=False, index=False)
        detail = pd.read_csv(detail_path)
        merged = rows.merge(detail, on="id_student", how="left")
        rows = merged
        ung_arr = rows["ungoverned_action"].to_numpy()
        est_p = rows["ungoverned_p_at_risk"].to_numpy(dtype=float)
        results["ungoverned"] = _summarize(oracle, ung_arr, est_p, model_p=p)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / f"bias_eval_w{cutoff:02d}.csv", index=False)
    summary_path = out_dir / f"bias_eval_w{cutoff:02d}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"per-student results: {out_dir / f'bias_eval_w{cutoff:02d}.csv'}")
    print(f"summary: {summary_path}")
    return results


def _summarize(oracle: np.ndarray, actions, est_p=None, model_p=None) -> dict:
    if isinstance(actions, list):
        actions = np.array(actions)
    valid = np.isin(actions, list(ACTION_LEVELS))
    act = np.array([a in ("outreach", "urgent_outreach") for a in actions])
    out = {
        "act_rate": float(act.mean()),
        "agreement_with_oracle": float((act == oracle).mean()),
        "over_intervention_rate": float((act & ~oracle).mean()),
        "under_intervention_rate": float((~act & oracle).mean()),
        "action_distribution": {
            level: int((actions == level).sum()) for level in ACTION_LEVELS
        },
        "n_errors": int((~valid).sum()),
    }
    if est_p is not None and model_p is not None:
        est = np.asarray(est_p, dtype=float)
        out["mean_abs_p_error_vs_model"] = float(np.nanmean(np.abs(est - model_p)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Intervention-bias evaluation")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-miss", type=float, default=4.0)
    parser.add_argument("--cost-intervention", type=float, default=1.0)
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the LLM agents, evaluate oracle vs governed policy only")
    args = parser.parse_args()
    run(
        n=args.n,
        cutoff=args.cutoff,
        seed=args.seed,
        cost_miss=args.cost_miss,
        cost_intervention=args.cost_intervention,
        run_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()
