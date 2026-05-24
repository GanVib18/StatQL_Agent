import re
import json
import numpy as np
import pandas as pd
from typing import Optional, TypedDict

from groq import Groq
from langgraph.graph import StateGraph, END

from data.data_layer import DataLayer
from stats.validator import StatValidator


class AgentState(TypedDict):
    question:     str
    schema:       str
    sql:          str
    query_result: Optional[pd.DataFrame]
    answer:       str
    error:        Optional[str]
    retries:      int
    stats:        Optional[dict]


class AnalyticsAgent:
    """
    LangGraph pipeline:
      parse_question → generate_sql → execute_query → synthesize_answer
                                 ↑___________ (retry on SQL error, max 3×)
    """
    MAX_RETRIES = 3
    MAX_ROWS    = 50

    def __init__(self, data_layer: DataLayer, groq_api_key: str, model: str):
        self.dl        = data_layer
        self.llm       = Groq(api_key=groq_api_key)
        self.model     = model
        self.validator = StatValidator()
        self.graph     = self._build_graph()

    # ── LLM helper ────────────────────────────────────────────────
    def _chat(self, system: str, user: str, max_tokens: int = 1024) -> str:
        r = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.1, max_tokens=max_tokens,
        )
        return r.choices[0].message.content.strip()

    # ── nodes ──────────────────────────────────────────────────────
    def _node_parse(self, state: AgentState) -> dict:
        return {"schema": self.dl.get_schema(), "retries": 0, "error": None}

    def _node_generate_sql(self, state: AgentState) -> dict:
        hint   = f"\n\nPrevious SQL error — fix it:\n{state['error']}" if state.get("error") else ""
        system = (
            "You are a DuckDB SQL expert. Output ONLY a valid DuckDB SELECT query in ```sql ... ``` fences."
            " Table: retail. InvoiceDate is TIMESTAMP. Revenue column exists (= Quantity * UnitPrice)."
            " Never use DROP/DELETE/INSERT/UPDATE."
        )
        raw   = self._chat(system, f"Schema:\n{state['schema']}\n\nQuestion: {state['question']}{hint}")
        match = re.search(r"```sql\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
        return {"sql": (match.group(1).strip() if match else raw.strip()), "error": None}

    def _node_execute(self, state: AgentState) -> dict:
        try:
            return {"query_result": self.dl.execute_query(state["sql"]), "error": None}
        except Exception as e:
            return {"query_result": None, "error": str(e), "retries": state.get("retries", 0) + 1}

    def _node_synthesize(self, state: AgentState) -> dict:
        df  = state.get("query_result")
        sc  = self.validator.validate(df, state["question"])

        if df is not None and not df.empty:
            preview = df.head(self.MAX_ROWS)
            result_str = preview.to_string(index=False)
            if len(df) > self.MAX_ROWS:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                summary  = {c: {"min": df[c].min(), "max": df[c].max(),
                                "mean": round(df[c].mean(), 2), "sum": round(df[c].sum(), 2)}
                            for c in num_cols[:3]}
                result_str += f"\n\n[{self.MAX_ROWS}/{len(df)} rows shown. Summary: {json.dumps(summary, default=str)}]"
        else:
            result_str = "No rows returned."

        system = (
            "You are a senior data analyst. Answer concisely (2–4 sentences) using the query results and stats."
            " Always cite confidence intervals or p-values where available. Flag non-significant results."
        )
        user = (f"Question: {state['question']}\n\nResults:\n{result_str}\n\n"
                f"Statistical context:\n{json.dumps(sc, default=str, indent=2)}")
        answer = self._chat(system, user, max_tokens=512)
        return {"answer": self._hallucination_guard(answer, df), "stats": sc}

    # ── hallucination guard ────────────────────────────────────────
    def _hallucination_guard(self, answer: str, df: Optional[pd.DataFrame]) -> str:
        if df is None or df.empty:
            return answer
        actual = [v for col in df.select_dtypes(include=[np.number]).columns
                    for v in df[col].dropna().tolist()]
        if not actual:
            return answer

        def matches(v: float) -> bool:
            if v <= 1: return True
            for a in actual:
                if abs(a) < 1e-9: continue
                for scale in [1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e9]:
                    if abs(v - a / scale) / (abs(a / scale) + 1e-9) < 0.05:
                        return True
            return False

        flagged = [n for n in re.findall(r"\b\d+(?:\.\d+)?\b", answer.replace(",", ""))[:15]
                   if not matches(float(n))]
        if flagged:
            answer += (f"\n\n⚠️ *Verify: {', '.join(flagged[:3])} could not be reconciled "
                       "with raw query output — please confirm against the underlying data.*")
        return answer

    # ── graph ──────────────────────────────────────────────────────
    def _route(self, state: AgentState) -> str:
        return "retry" if (state.get("error") and state.get("retries", 0) < self.MAX_RETRIES) else "synthesize"

    def _build_graph(self):
        g = StateGraph(AgentState)
        for name, fn in [("parse_question",    self._node_parse),
                         ("generate_sql",      self._node_generate_sql),
                         ("execute_query",     self._node_execute),
                         ("synthesize_answer", self._node_synthesize)]:
            g.add_node(name, fn)
        g.set_entry_point("parse_question")
        g.add_edge("parse_question", "generate_sql")
        g.add_edge("generate_sql",   "execute_query")
        g.add_conditional_edges("execute_query", self._route,
                                {"retry": "generate_sql", "synthesize": "synthesize_answer"})
        g.add_edge("synthesize_answer", END)
        return g.compile()

    def ask(self, question: str) -> dict:
        out = self.graph.invoke({
            "question": question, "schema": "", "sql": "",
            "query_result": None, "answer": "", "error": None, "retries": 0, "stats": None,
        })
        return {k: out.get(k) for k in ["question", "sql", "answer", "stats", "retries", "error"]}
