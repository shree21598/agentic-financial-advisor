"""Run the ADK agent programmatically (used by the eval harness)."""
from __future__ import annotations

import uuid

from google.genai import types
from google.adk.runners import InMemoryRunner

from advisor.adk_agent.agent import TOOL_LOG, root_agent

_SPECIALISTS = {"research_agent", "planning_agent", "market_agent"}


class ADKAdvisor:
    def __init__(self):
        self.runner = InMemoryRunner(agent=root_agent, app_name="financial_advisor")

    async def ask(self, query: str) -> dict:
        TOOL_LOG.clear()
        uid, sid = "eval", str(uuid.uuid4())
        await self.runner.session_service.create_session(app_name="financial_advisor", user_id=uid, session_id=sid)
        answer = ""
        msg = types.Content(role="user", parts=[types.Part(text=query)])
        async for ev in self.runner.run_async(user_id=uid, session_id=sid, new_message=msg):
            if ev.is_final_response() and ev.content and ev.content.parts:
                answer = "".join(p.text or "" for p in ev.content.parts)
        calls = [c for c in TOOL_LOG if c["tool"] not in _SPECIALISTS]
        return {
            "answer": answer,
            "tool_calls": [{"tool": c["tool"], "args": c["args"], "ok": "error" not in (c["result"] or {})} for c in calls],
            "evidence": [c for c in calls if c["result"] is not None],
        }
