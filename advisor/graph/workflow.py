"""LangGraph orchestration of the advisor.

    guardrail ──blocked──▶ finalize
        │
        ▼
      agent ◀──────────┐            (Gemini with function calling)
        │ tool calls?  │
        ├──yes──▶ tools ┘           (MCP or in-process executor, parallel)
        │ no
        ▼
    grounding_check ──score<τ & revisions left──▶ agent (with feedback)
        │
        ▼
     finalize                        (compliance disclaimer)
"""
from __future__ import annotations

import asyncio
import operator
import time
from typing import Annotated, Any, TypedDict

from google.genai import types
from langgraph.graph import END, StateGraph

from advisor.config import settings
from advisor.graph.compliance import check_input, finalize
from advisor.graph.grounding import judge_groundedness
from advisor.llm import ChatModel, get_chat_model
from advisor.prompts import ADVISOR_SYSTEM


class AdvisorState(TypedDict, total=False):
    query: str
    contents: list[types.Content]
    tool_calls: Annotated[list[dict], operator.add]
    evidence: Annotated[list[dict], operator.add]
    rounds: int
    revisions: int
    answer: str
    blocked: bool
    groundedness: dict


def _text(c: types.Content) -> str:
    return "".join(p.text or "" for p in c.parts or [] if not getattr(p, "thought", False))


def build_graph(executor, model: ChatModel | None = None, inline_grounding: bool = False):
    cfg = settings()
    model = model or get_chat_model()
    decl_cache: list[dict] = []

    async def declarations():
        if not decl_cache:
            decl_cache.extend(await executor.declarations())
        return decl_cache

    async def guardrail(state: AdvisorState):
        refusal = check_input(state["query"])
        if refusal:
            return {"blocked": True, "answer": refusal}
        return {
            "blocked": False,
            "rounds": 0,
            "revisions": 0,
            "contents": [types.Content(role="user", parts=[types.Part(text=state["query"])])],
        }

    async def agent(state: AdvisorState):
        tools = await declarations() if state["rounds"] < cfg.max_tool_rounds else None
        out = await asyncio.to_thread(model.generate, ADVISOR_SYSTEM, state["contents"], tools)
        return {"contents": state["contents"] + [out], "answer": _text(out)}

    async def run_tools(state: AdvisorState):
        calls = [p.function_call for p in state["contents"][-1].parts if p.function_call]

        async def one(fc):
            args = dict(fc.args or {})
            t0 = time.perf_counter()
            try:
                res, ok = await executor.call(fc.name, args), True
            except Exception as e:  # surface tool errors back to the model
                res, ok = {"error": str(e)}, False
            return fc, args, res, ok, (time.perf_counter() - t0) * 1000

        results = await asyncio.gather(*(one(fc) for fc in calls))
        parts, log, ev = [], [], []
        for fc, args, res, ok, ms in results:
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": res}))
            log.append({"tool": fc.name, "args": args, "ok": ok, "ms": round(ms, 1)})
            if ok:
                ev.append({"tool": fc.name, "args": args, "result": res})
        return {
            "contents": state["contents"] + [types.Content(role="user", parts=parts)],
            "tool_calls": log,
            "evidence": ev,
            "rounds": state["rounds"] + 1,
        }

    async def grounding_check(state: AdvisorState):
        g = await asyncio.to_thread(judge_groundedness, state.get("answer", ""), state.get("evidence", []))
        update: dict[str, Any] = {"groundedness": g}
        if g["score"] < cfg.groundedness_threshold and state["revisions"] < cfg.max_revisions:
            bad = [c["claim"] for c in g["claims"] if not c.get("supported")][:5]
            fb = (
                "Revise your answer. These claims are not supported by tool outputs: "
                + "; ".join(bad)
                + ". Remove them or call tools to support them. Keep supported content and citations."
            )
            update["contents"] = state["contents"] + [types.Content(role="user", parts=[types.Part(text=fb)])]
            update["revisions"] = state["revisions"] + 1
        return update

    async def finalize_node(state: AdvisorState):
        return {"answer": finalize(state.get("answer", ""))}

    def after_agent(state: AdvisorState):
        if any(p.function_call for p in state["contents"][-1].parts or []):
            return "tools"
        return "grounding_check" if inline_grounding else "finalize"

    def after_check(state: AdvisorState):
        return "agent" if state["contents"][-1].role == "user" else "finalize"

    g = StateGraph(AdvisorState)
    g.add_node("guardrail", guardrail)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("grounding_check", grounding_check)
    g.add_node("finalize", finalize_node)
    g.set_entry_point("guardrail")
    g.add_conditional_edges("guardrail", lambda s: "finalize" if s["blocked"] else "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "grounding_check", "finalize"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("grounding_check", after_check, ["agent", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


async def ask(query: str, executor=None, inline_grounding: bool = False) -> AdvisorState:
    from advisor.tools.executor import get_executor

    async with (executor or get_executor()) as ex:
        graph = build_graph(ex, inline_grounding=inline_grounding)
        return await graph.ainvoke({"query": query}, {"recursion_limit": 30})


if __name__ == "__main__":  # python -m advisor.graph.workflow "question"
    import sys

    st = asyncio.run(ask(" ".join(sys.argv[1:]) or "What is the 4% rule?"))
    print(st["answer"])
    print("\ntools:", [(c["tool"], c["ok"]) for c in st.get("tool_calls", [])])
