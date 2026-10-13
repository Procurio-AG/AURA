"""Governed advisor: LLM retrieves and explains, calibrated LR decides the action."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from src.agent.report import EvidenceItem, InterventionReport, RationaleFill
from src.agent.tools import (
    DB_PATH,
    DEFAULT_CUTOFF,
    MODELS_DIR,
    ModelBackend,
    action_from_p,
    make_llm,
    make_model_tools,
    make_sql_tools,
)

GOVERNED_PROMPT = """You are a student support advisor agent operating on the OULAD database.

Rules you MUST follow:
1. Never judge a student's risk yourself. All risk judgements MUST come from the
   predict_risk tool, which returns calibrated probabilities.
2. Investigate with the SQL tools (weekly_clicks, weekly_activity, features_wXX,
   studentAssessment, etc.) to gather behavioural evidence. Use aggregated queries
   (COUNT, AVG, GROUP BY) rather than SELECT * on large tables.
3. When recommending an action for a specific student, you MUST call
   write_intervention_report. It computes the action from the model probability;
   you supply the evidence, rationale and caveats.
4. Every evidence item must state where it came from (a SQL query result or the
   model output). Do not invent facts.
5. Communicate uncertainty: mention data sparsity at early weeks and that the model
   is probabilistic.

Answer instructor questions concisely, citing the tools you used."""


def _report_tool(backend: ModelBackend):
    from langchain_core.tools import tool

    @tool
    def write_intervention_report(
        student_id: int,
        week: int,
        evidence: list[str],
        action_rationale: str,
        uncertainty_caveats: list[str],
    ) -> str:
        """Write the final intervention report for one student. The recommended
        action is computed from the calibrated model probability, not from your
        own judgement. evidence: list of cited facts (prefix each with 'sql:' or
        'model:'). action_rationale: why this action. uncertainty_caveats:
        limitations."""
        risk = backend.risk(student_id, week)
        if not risk.get("found"):
            return json.dumps({"error": risk.get("error")})
        reports = []
        for entry in risk["students"]:
            p = entry["p_at_risk"]
            ev = []
            for item in evidence:
                source, _, fact = item.partition(":")
                ev.append(EvidenceItem(
                    source="sql" if source.strip() == "sql" else "model",
                    fact=fact.strip(),
                ))
            reports.append(InterventionReport(
                student_id=int(student_id),
                cutoff_week=int(week),
                p_at_risk=p,
                risk_band=entry["risk_band"],
                evidence=ev,
                recommended_action=action_from_p(p),
                action_rationale=action_rationale,
                uncertainty_caveats=uncertainty_caveats,
            ).model_dump())
        return json.dumps(reports, indent=2)

    return write_intervention_report


class GovernedAdvisor:
    def __init__(self, db_path: Path = DB_PATH, models_dir: Path = MODELS_DIR,
                 cutoff: int = DEFAULT_CUTOFF, llm=None):
        self.backend = ModelBackend(cutoff, db_path, models_dir)
        self.llm = llm or make_llm()
        self.tools = (
            make_sql_tools(db_path)
            + make_model_tools(self.backend)
            + [_report_tool(self.backend)]
        )

    def _agent(self):
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(self.llm, self.tools, prompt=GOVERNED_PROMPT)

    def chat(self, question: str) -> str:
        result = self._agent().invoke(
            {"messages": [("user", question)]}, config={"recursion_limit": 40}
        )
        return result["messages"][-1].content

    def report_for_student(self, student_id: int, week: Optional[int] = None) -> list[dict]:
        week = week if week is not None else self.backend.cutoff
        risk = self.backend.risk(student_id, week)
        if not risk.get("found"):
            return [{"error": risk.get("error")}]
        facts = json.dumps(risk["students"], indent=2)
        fill = self.llm.with_structured_output(RationaleFill).invoke(
            [
                (
                    "system",
                    "You write intervention report narratives for at-risk students. "
                    "Use only the facts given. Be specific and concise.",
                ),
                (
                    "human",
                    f"Model risk assessment for student {student_id} at week {week}:\n"
                    f"{facts}\n\nWrite action_rationale (2-3 sentences, referencing the "
                    "drivers) and uncertainty_caveats (1-3 items).",
                ),
            ]
        )
        reports = []
        for entry in risk["students"]:
            p = entry["p_at_risk"]
            drivers = entry["top_drivers"][:3]
            evidence = [
                EvidenceItem(
                    source="model",
                    fact=f"p(at-risk) = {p:.3f} at week {week} "
                    f"({entry['code_module']}/{entry['code_presentation']})",
                )
            ]
            for d in drivers:
                evidence.append(EvidenceItem(
                    source="model",
                    fact=f"{d['feature']} {d['direction']} "
                    f"(contribution {d['contribution']:.2f})",
                ))
            reports.append(InterventionReport(
                student_id=int(student_id),
                cutoff_week=int(week),
                p_at_risk=p,
                risk_band=entry["risk_band"],
                evidence=evidence,
                recommended_action=action_from_p(p),
                action_rationale=fill.action_rationale,
                uncertainty_caveats=fill.uncertainty_caveats,
            ).model_dump())
        return reports


def sample_test_students(cutoff: int, n: int, seed: int = 42, db_path: Path = DB_PATH):
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        split = pd.read_sql_query(
            f"SELECT * FROM split_w{cutoff:02d} WHERE is_test = 1", con
        )
    return split.sample(n=min(n, len(split)), random_state=seed)


if __name__ == "__main__":
    import sys

    import pandas as pd

    advisor = GovernedAdvisor()
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Which BBB students went quiet before TMA-2, and who needs outreach this week?"
    )
    print(advisor.chat(question))
