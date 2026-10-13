#!/usr/bin/env python3
"""Connection & Simultaneous Comparison Script: Gemma 26B vs Gemma 31B.

Connects to Google AI Studio via OpenAI-compatible endpoint, runs both models
concurrently on the Tool-Governed Early Warning System (EWS), and outputs
comparative evaluation metrics.
"""

from __future__ import annotations

import os
import sys
import time
import json
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "final") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "final"))

# Load environment
load_dotenv(PROJECT_ROOT / ".env")

from langchain_openai import ChatOpenAI
from final.src.agent.tools import (
    ModelBackend,
    action_from_p,
    risk_band_from_p,
    act_threshold,
    DEFAULT_CUTOFF,
    DEFAULT_COST_MISS,
    DEFAULT_COST_INTERVENTION,
)
from final.src.agent.advisor import GovernedAdvisor
from final.src.agent.advisor_baseline import UngovernedAdvisor


def get_llm(model_name: str, api_key: str, base_url: str) -> ChatOpenAI:
    """Instantiate OpenAI-compatible Chat client pointing to Google AI Studio."""
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0,
        timeout=120,
        max_retries=2,
    )


def evaluate_student(sid: int, advisor: Any, cutoff: int = 8) -> Dict[str, Any]:
    """Run single-student report generation and measure latency."""
    t0 = time.time()
    try:
        reports = advisor.report_for_student(int(sid), cutoff)
        latency = time.time() - t0
        report = reports[0] if isinstance(reports, list) else reports
        return {
            "student_id": sid,
            "status": "success",
            "latency": latency,
            "p_at_risk": float(report.get("p_at_risk", np.nan)),
            "recommended_action": report.get("recommended_action", "none"),
            "risk_band": report.get("risk_band", "low"),
            "evidence_count": len(report.get("evidence", [])),
            "caveats_count": len(report.get("uncertainty_caveats", [])),
            "rationale": report.get("action_rationale", ""),
        }
    except Exception as e:
        latency = time.time() - t0
        return {
            "student_id": sid,
            "status": "error",
            "error": str(e),
            "latency": latency,
            "p_at_risk": np.nan,
            "recommended_action": "error",
            "risk_band": "error",
            "evidence_count": 0,
            "caveats_count": 0,
            "rationale": "",
        }


def run_model_eval(model_name: str, api_key: str, base_url: str,
                   students: list[int], db_path: Path, models_dir: Path,
                   governed: bool = True) -> Dict[str, Any]:
    """Run evaluation for a specific model over cohort students."""
    llm = get_llm(model_name, api_key, base_url)
    advisor_cls = GovernedAdvisor if governed else UngovernedAdvisor
    advisor = advisor_cls(db_path=db_path, models_dir=models_dir, cutoff=8, llm=llm)

    results = []
    for sid in students:
        res = evaluate_student(sid, advisor, cutoff=8)
        results.append(res)

    return {
        "model": model_name,
        "governed": governed,
        "results": results,
    }


def main():
    print("=" * 80)
    print("GOOGLE AI STUDIO DUAL-MODEL EVALUATION: GEMMA 26B vs GEMMA 31B")
    print("=" * 80)

    # Read configuration from .env
    base_url = os.getenv("GOOGLE_AI_STUDIO_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    key_26b = os.getenv("GEMINI_API_KEY_GEMMA_26B") or os.getenv("GEMINI_API_KEY")
    key_31b = os.getenv("GEMINI_API_KEY_GEMMA_31B") or os.getenv("GEMINI_API_KEY")
    model_26b = os.getenv("MODEL_GEMMA_26B", "gemma-4-26b-a4b-it")
    model_31b = os.getenv("MODEL_GEMMA_31B", "gemma-4-31b-it")

    db_path = PROJECT_ROOT / "oulad.db"
    models_dir = PROJECT_ROOT / "models"

    print(f"[*] Base URL:           {base_url}")
    print(f"[*] Model 1 (Gemma 26B): {model_26b}")
    print(f"[*] Model 2 (Gemma 31B): {model_31b}")

    # Check for placeholder keys
    if not key_26b or key_26b == "your_google_ai_studio_key_here":
        print("\n[!] NOTICE: Google AI Studio API key not found or still set to placeholder in .env.")
        print("[!] Please open '.env' and paste your Gemini API key from https://aistudio.google.com/app/apikey into:")
        print("      GEMINI_API_KEY_GEMMA_26B=...")
        print("      GEMINI_API_KEY_GEMMA_31B=...")
        print("\n[✓] The script and environment are fully configured and ready to run once the key is pasted.")
        sys.exit(0)

    # Initialize backend & sample students
    backend = ModelBackend(cutoff=8, db_path=db_path, models_dir=models_dir)
    test_students = backend.features[backend.features["code_presentation"] == "2014J"]["id_student"].drop_duplicates().head(10).tolist()
    print(f"[*] Evaluation sample: {len(test_students)} students from 2014J held-out test cohort")

    print("\n[*] Launching simultaneous evaluation of Gemma 26B and Gemma 31B in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_26b = executor.submit(
            run_model_eval, model_26b, key_26b, base_url, test_students, db_path, models_dir, True
        )
        future_31b = executor.submit(
            run_model_eval, model_31b, key_31b, base_url, test_students, db_path, models_dir, True
        )

        res_26b = future_26b.result()
        res_31b = future_31b.result()

    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION RESULTS SCORECARD")
    print("=" * 80)

    # Compute comparative metrics
    df_26b = pd.DataFrame(res_26b["results"])
    df_31b = pd.DataFrame(res_31b["results"])

    succ_26b = df_26b[df_26b["status"] == "success"]
    succ_31b = df_31b[df_31b["status"] == "success"]

    metrics_table = [
        ("Evaluated Cohort Size", f"{len(test_students)}", f"{len(test_students)}"),
        ("Successful Report Generations", f"{len(succ_26b)} / {len(df_26b)}", f"{len(succ_31b)} / {len(df_31b)}"),
        ("Mean Latency per Report", f"{succ_26b['latency'].mean():.2f} s", f"{succ_31b['latency'].mean():.2f} s"),
        ("Mean Evidence Items Cited", f"{succ_26b['evidence_count'].mean():.1f}", f"{succ_31b['evidence_count'].mean():.1f}"),
        ("Mean Uncertainty Caveats", f"{succ_26b['caveats_count'].mean():.1f}", f"{succ_31b['caveats_count'].mean():.1f}"),
        ("Action: Urgent Outreach %", f"{(succ_26b['recommended_action'] == 'urgent_outreach').mean()*100:.1f}%", f"{(succ_31b['recommended_action'] == 'urgent_outreach').mean()*100:.1f}%"),
        ("Action: Outreach %", f"{(succ_26b['recommended_action'] == 'outreach').mean()*100:.1f}%", f"{(succ_31b['recommended_action'] == 'outreach').mean()*100:.1f}%"),
        ("Action: Monitor %", f"{(succ_26b['recommended_action'] == 'monitor').mean()*100:.1f}%", f"{(succ_31b['recommended_action'] == 'monitor').mean()*100:.1f}%"),
    ]

    header = f"{'Metric':<35} | {'Gemma 26B (MoE)':<20} | {'Gemma 31B (Dense)':<20}"
    print(header)
    print("-" * len(header))
    for m, v1, v2 in metrics_table:
        print(f"{m:<35} | {v1:<20} | {v2:<20}")

    print("=" * 80)
    print("[✓] Dual-model evaluation complete!")


if __name__ == "__main__":
    main()
