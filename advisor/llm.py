"""Model layer: Gemini on Vertex AI, plus a deterministic offline stub for CI.

Both implement ``ChatModel.generate(system, contents, tools) -> types.Content`` using
google-genai types, so the LangGraph agent loop is identical online and offline.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Protocol

from google.genai import types

from advisor.config import settings


@lru_cache(maxsize=1)
def genai_client():
    from google import genai

    s = settings()
    if not s.project:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT (or ADVISOR_OFFLINE=1 for the offline stub).")
    return genai.Client(vertexai=True, project=s.project, location=s.location)


class ChatModel(Protocol):
    def generate(
        self, system: str, contents: list[types.Content], tools: list[dict] | None = None, json_mode: bool = False
    ) -> types.Content: ...


class GeminiChat:
    def __init__(self, model: str | None = None, temperature: float = 0.1):
        self.model = model or settings().model
        self.temperature = temperature

    def generate(self, system, contents, tools=None, json_mode=False):
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if tools:
            cfg.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]
                        )
                        for t in tools
                    ]
                )
            ]
        if json_mode:
            cfg.response_mime_type = "application/json"
        resp = genai_client().models.generate_content(model=self.model, contents=contents, config=cfg)
        cand = resp.candidates[0] if resp.candidates else None
        if cand is None or cand.content is None:
            return types.Content(role="model", parts=[types.Part(text="")])
        return cand.content


# --------------------------------------------------------------------------- offline stub
_NUM = r"\$?([\d,]+(?:\.\d+)?)\s*(k)?"
TICKERS = ["VTI", "VXUS", "QQQ", "BND", "GLD", "VNQ"]


def _money(s: str) -> float:
    m = re.search(_NUM, s, re.I)
    return float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1) if m else 0.0


def _pct(text: str, default: float) -> float:
    m = re.search(r"([\d.]+)\s*%", text)
    return float(m.group(1)) if m else default


def _int_before(text: str, word: str, default: int) -> int:
    m = re.search(rf"(\d+)[\s-]*{word}", text, re.I)
    return int(m.group(1)) if m else default


def offline_plan(q: str) -> list[tuple[str, dict]]:
    """Keyword/regex tool planner used only by the offline stub."""
    ql = q.lower()
    calls: list[tuple[str, dict]] = []
    tick = [t for t in TICKERS if re.search(rf"\b{t}\b", q)]
    if len(tick) >= 2 or (tick and "portfolio" in ql):
        weights = {}
        for t in tick:
            m = re.search(rf"(\d+)\s*%\s*{t}|{t}\s*(\d+)\s*%", q)
            weights[t] = float((m.group(1) or m.group(2)) if m else 1)
        calls.append(("portfolio_metrics", {"holdings": weights}))
    elif tick:
        calls.append(("get_price_history", {"ticker": tick[0]}))
    if re.search(r"mortgage|loan|monthly payment", ql):
        calls.append(("loan_amortization", {
            "principal": _money(q), "annual_rate_pct": _pct(q, 6.0), "years": _int_before(q, "year", 30)}))
    elif "retire" in ql and re.search(r"\d+", q):
        ages = [int(a) for a in re.findall(r"\b(\d{2})\b", q)]
        amounts = [float(a.replace(",", "")) for a in re.findall(r"\$([\d,]+)", q)]
        calls.append(("retirement_projection", {
            "current_age": ages[0] if ages else 30, "retirement_age": ages[1] if len(ages) > 1 else 65,
            "current_savings": amounts[0] if amounts else 0, "monthly_contribution": amounts[1] if len(amounts) > 1 else 0,
            "expected_return_pct": _pct(q, 6.0), "annual_spending_in_retirement": amounts[2] if len(amounts) > 2 else 0}))
    elif re.search(r"grow|be worth|future value|invest \$", ql) and "$" in q:
        amounts = [float(a.replace(",", "")) for a in re.findall(r"\$([\d,]+)", q)]
        calls.append(("compound_growth", {
            "principal": amounts[0] if amounts else 0, "annual_rate_pct": _pct(q, 7.0),
            "years": _int_before(q, "year", 10), "monthly_contribution": amounts[1] if len(amounts) > 1 else 0}))
    if "emergency fund" in ql and "$" in q:
        amounts = [float(a.replace(",", "")) for a in re.findall(r"\$([\d,]+)", q)]
        calls.append(("emergency_fund_target", {
            "monthly_expenses": amounts[0], "months": _int_before(q, "month", 6),
            "current_savings": amounts[1] if len(amounts) > 1 else 0}))
    if re.search(r"50/30/20|budget", ql) and "$" in q:
        calls.append(("budget_50_30_20", {"monthly_net_income": _money(q)}))
    if re.search(r"avalanche|snowball", ql) and "$" in q:
        debts = [{"name": n, "balance": float(b.replace(",", "")), "apr_pct": float(a), "min_payment": float(mp)}
                 for n, b, a, mp in re.findall(r"(\w+) \$([\d,]+) at ([\d.]+)% \(min \$(\d+)\)", q)]
        budget = re.search(r"\$([\d,]+)\s*(?:/|per|a)\s*month", q)
        if debts and budget:
            calls.append(("debt_payoff_plan", {"debts": debts, "monthly_budget": float(budget.group(1).replace(",", "")),
                                               "strategy": "snowball" if "snowball" in ql else "avalanche"}))
    conceptual = re.search(r"what|why|how|explain|should|difference|pros|cons|vs\.?|better|rule|mean", ql)
    if conceptual or not calls:
        calls.append(("search_knowledge_base", {"query": q, "top_k": 4}))
    return calls


class OfflineChat:
    """Deterministic stand-in for Gemini. Exercises the full graph without network access.

    Metrics produced with this model are smoke-test numbers only -- not model quality.
    """

    model = "offline-stub"

    def generate(self, system, contents, tools=None, json_mode=False):
        if json_mode:  # used by the groundedness judge
            return types.Content(role="model", parts=[types.Part(text=_offline_judge(contents))])
        last = contents[-1]
        responses = [p.function_response for p in last.parts or [] if p.function_response]
        user_q = next((p.text for c in contents if c.role == "user" for p in c.parts if p.text), "")
        if not responses and tools and not any(p.function_call for c in contents for p in c.parts or []):
            names = {t["name"] for t in tools}
            parts = [types.Part(function_call=types.FunctionCall(name=n, args=a))
                     for n, a in offline_plan(user_q) if n in names]
            if parts:
                return types.Content(role="model", parts=parts)
        return types.Content(role="model", parts=[types.Part(text=_offline_answer(contents))])


def _offline_answer(contents: list[types.Content]) -> str:
    lines, cites = [], []
    for c in contents:
        for p in c.parts or []:
            fr = p.function_response
            if not fr:
                continue
            res = fr.response.get("result", fr.response)
            if fr.name == "search_knowledge_base" and isinstance(res, dict):
                for h in res.get("results", [])[:2]:
                    body = h["text"].split(": ", 1)[-1]
                    sent = re.split(r"(?<=[.!?])\s", body)
                    lines.append(" ".join(sent[:2]) + f" [{h['source']}]")
            elif isinstance(res, dict):
                kv = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in res.items()
                               if isinstance(v, (int, float, str, bool)) and v not in ("",))
                lines.append(f"From {fr.name}: {kv}.")
    return "\n".join(lines) or "I could not find relevant information."


def _offline_judge(contents: list[types.Content]) -> str:
    """Lexical-overlap claim checker so offline evals exercise the judge path."""
    from advisor.rag.embeddings import tokenize

    text = contents[-1].parts[0].text
    ctx, _, ans = text.partition("ANSWER:")
    ctx_toks = set(tokenize(ctx))
    claims = [s for s in re.split(r"(?<=[.!?])\s+|\n", ans) if len(tokenize(s)) >= 3]
    verdicts = []
    for c in claims:
        toks = set(tokenize(c))
        verdicts.append({"claim": c[:200], "supported": len(toks & ctx_toks) / max(len(toks), 1) >= 0.7})
    return json.dumps({"claims": verdicts})


def get_chat_model(model: str | None = None) -> ChatModel:
    return OfflineChat() if settings().offline else GeminiChat(model)
