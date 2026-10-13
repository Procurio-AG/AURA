"""Model backend + LangChain tools (predict_risk, predict_next_score, SQL toolkit)."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features import KEYS
from src.models import load_bundle, top_drivers

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "oulad.db"
MODELS_DIR = ROOT / "models"
DEFAULT_CUTOFF = 8

DEFAULT_COST_MISS = 4.0
DEFAULT_COST_INTERVENTION = 1.0


def act_threshold(cost_miss: float = DEFAULT_COST_MISS,
                  cost_intervention: float = DEFAULT_COST_INTERVENTION) -> float:
    """Act iff E[cost | act] < E[cost | don't act] => p >= ci / (cm + ci)."""
    return cost_intervention / (cost_miss + cost_intervention)


def action_from_p(p: float, cost_miss: float = DEFAULT_COST_MISS,
                  cost_intervention: float = DEFAULT_COST_INTERVENTION) -> str:
    t = act_threshold(cost_miss, cost_intervention)
    if p >= (1.0 + t) / 2.0:
        return "urgent_outreach"
    if p >= t:
        return "outreach"
    if p >= t / 2.0:
        return "monitor"
    return "none"


def risk_band_from_p(p: float, cost_miss: float = DEFAULT_COST_MISS,
                     cost_intervention: float = DEFAULT_COST_INTERVENTION) -> str:
    t = act_threshold(cost_miss, cost_intervention)
    if p >= t:
        return "high"
    if p >= t / 2.0:
        return "moderate"
    return "low"


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


class ModelBackend:
    def __init__(self, cutoff: int = DEFAULT_CUTOFF, db_path: Path = DB_PATH,
                 models_dir: Path = MODELS_DIR):
        self.cutoff = cutoff
        self.bundle = load_bundle(models_dir / f"agent_bundle_w{cutoff:02d}.joblib")
        self.feature_cols = self.bundle["feature_cols"]
        with _connect_ro(db_path) as con:
            self.features = pd.read_sql_query(
                f"SELECT * FROM features_w{cutoff:02d}", con
            )
        self.features = self.features.set_index(KEYS, drop=False)

    def _module_rows(self, student_id: int):
        rows = self.features[self.features["id_student"] == int(student_id)]
        return rows

    def risk(self, student_id: int, week: Optional[int] = None) -> dict:
        week = week if week is not None else self.cutoff
        rows = self._module_rows(student_id)
        if rows.empty:
            return {"found": False,
                    "error": f"student {student_id} not found in features_w{week:02d}"}
        clf = self.bundle["classifier"]
        results = []
        for idx, row in rows.iterrows():
            x = row[self.feature_cols].to_frame().T
            p = float(clf.predict_proba(x)[0, 1])
            drivers = top_drivers(clf, x.iloc[0], self.feature_cols)
            results.append({
                "found": True,
                "code_module": row["code_module"],
                "code_presentation": row["code_presentation"],
                "cutoff_week": week,
                "p_at_risk": p,
                "risk_band": risk_band_from_p(p),
                "model": "logistic_regression (calibrated via logit link)",
                "top_drivers": drivers,
            })
        return {"found": True, "students": results}

    def next_score(self, student_id: int) -> dict:
        rows = self._module_rows(student_id)
        if rows.empty:
            return {"found": False,
                    "error": f"student {student_id} not found in features_w{self.cutoff:02d}"}
        reg = self.bundle["regressor"]
        results = []
        for idx, row in rows.iterrows():
            x = row[self.feature_cols].to_frame().T
            pred = float(reg.predict(x)[0])
            results.append({
                "code_module": row["code_module"],
                "code_presentation": row["code_presentation"],
                "predicted_next_score": max(0.0, min(100.0, pred)),
            })
        return {"found": True, "predictions": results}


def _risk_to_json(student_id: int, week: int, backend: ModelBackend) -> str:
    return json.dumps(backend.risk(student_id, week), indent=2)


def make_model_tools(backend: ModelBackend):
    from langchain_core.tools import tool

    @tool
    def predict_risk(student_id: int, week: int) -> str:
        """Predict the calibrated probability that a student is at-risk
        (Withdrawn or Fail) at the given cutoff week, with top feature drivers.
        Use this for ALL risk judgements; never estimate risk yourself."""
        return _risk_to_json(student_id, week, backend)

    @tool
    def predict_next_score(student_id: int) -> str:
        """Predict the student's next assessment score (0-100) at the current
        cutoff using the ridge regressor."""
        return json.dumps(backend.next_score(student_id), indent=2)

    return [predict_risk, predict_next_score]


def make_sql_tools(db_path: Path = DB_PATH, max_rows: int = 100):
    import re

    from langchain_core.tools import tool as _tool
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool
    from langchain_community.utilities import SQLDatabase
    from langchain_community.tools.sql_database.tool import (
        InfoSQLDatabaseTool,
        ListSQLDatabaseTool,
    )

    def _connect():
        uri = f"file:{db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)

    engine = create_engine("sqlite://", creator=_connect, poolclass=NullPool)
    db = SQLDatabase(engine)
    table_names = db.get_usable_table_names()

    @_tool
    def sql_query(query: str) -> str:
        """Execute a read-only SQL query against the OULAD SQLite database.
        Queries without an explicit LIMIT are automatically capped at 100 rows.
        Prefer the aggregated tables (weekly_clicks, weekly_activity) over the
        raw studentVle table (10.6M rows)."""
        q = query.strip().rstrip(";")
        if not re.search(r"\blimit\b", q, re.IGNORECASE):
            q = f"SELECT * FROM ({q}) AS _sub LIMIT {max_rows}"
        con = _connect_ro(db_path)
        try:
            cur = con.execute(q)
            if cur.description is None:
                return json.dumps({"rows": []})
            rows = cur.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
            cols = [d[0] for d in cur.description]
            payload = [dict(zip(cols, r)) for r in rows]
            for r in payload:
                for k, v in list(r.items()):
                    if isinstance(v, bytes):
                        r[k] = v.decode("utf-8", "replace")
            return json.dumps(
                {"n_rows": len(payload), "truncated": truncated, "rows": payload},
                default=str,
            )
        except sqlite3.Error as exc:
            return json.dumps({"error": f"SQL error: {exc}"})
        finally:
            con.close()

    return [
        ListSQLDatabaseTool(db=db),
        InfoSQLDatabaseTool(db=db, table_names=table_names),
        sql_query,
    ]


def make_llm():
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; the agent layer requires it."
        )
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=os.environ.get("AURA_AGENT_MODEL", "openai/gpt-4o-mini"),
        temperature=0,
        timeout=120,
        max_retries=2,
    )
