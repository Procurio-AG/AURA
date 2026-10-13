"""Ungoverned baseline advisor: LLM judges risk freely from SQL evidence (the bias-prone variant)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.agent.report import InterventionReport
from src.agent.tools import (
    DB_PATH,
    DEFAULT_CUTOFF,
    MODELS_DIR,
    ModelBackend,
    make_llm,
    make_sql_tools,
)

UNGOVERNED_PROMPT = """You are a student support advisor agent operating on the OULAD database.

You have SQL tools to investigate student behaviour (weekly_clicks, weekly_activity,
features_wXX, studentAssessment, etc.). Always use the aggregated tables instead of
raw studentVle, and write queries with aggregates (COUNT, AVG, GROUP BY) rather than
SELECT * on large tables. You do NOT have a predictive model.
Assess each student's risk of withdrawing or failing yourself from the data you
retrieve, and decide on your own what action to recommend:
- none: no action needed
- monitor: keep an eye on the student
- outreach: contact the student
- urgent_outreach: contact the student immediately

Answer instructor questions concisely, citing the queries you ran."""


class UngovernedAdvisor:
    def __init__(self, db_path: Path = DB_PATH, models_dir: Path = MODELS_DIR,
                 cutoff: int = DEFAULT_CUTOFF, llm=None):
        self.backend = ModelBackend(cutoff, db_path, models_dir)
        self.llm = llm or make_llm()
        self.tools = make_sql_tools(db_path)

    def _agent(self):
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(self.llm, self.tools, prompt=UNGOVERNED_PROMPT)

    def chat(self, question: str) -> str:
        result = self._agent().invoke({"messages": [("user", question)]})
        return result["messages"][-1].content

    def report_for_student(self, student_id: int, week: Optional[int] = None) -> list[dict]:
        week = week if week is not None else self.backend.cutoff
        agent = self._agent()
        result = agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Assess student {student_id} (week {week} of the module). Investigate "
                        "their engagement with SQL, then state your risk estimate and "
                        "recommended action.",
                    )
                ]
            },
            config={"recursion_limit": 30},
        )
        transcript = "\n".join(
            m.content for m in result["messages"] if hasattr(m, "content") and m.content
        )
        report = self.llm.with_structured_output(InterventionReport).invoke(
            [
                (
                    "system",
                    "Fill an InterventionReport from your investigation. p_at_risk is "
                    "YOUR OWN estimate between 0 and 1. recommended_action is YOUR OWN "
                    "decision. evidence must cite your SQL findings.",
                ),
                ("human", transcript),
            ]
        )
        return [report.model_dump()]


if __name__ == "__main__":
    import sys

    advisor = UngovernedAdvisor()
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Which BBB students went quiet before TMA-2, and who needs outreach this week?"
    )
    print(advisor.chat(question))
