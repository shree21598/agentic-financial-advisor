"""LLM-as-judge groundedness: fraction of answer claims supported by the evidence."""
from __future__ import annotations

import json
import re

from google.genai import types

from advisor.graph.compliance import DISCLAIMER
from advisor.llm import ChatModel, get_chat_model
from advisor.prompts import JUDGE_SYSTEM
from advisor.config import settings


def format_evidence(evidence: list[dict]) -> str:
    blocks = []
    for e in evidence:
        res = e["result"]
        if e["tool"] == "search_knowledge_base" and isinstance(res, dict):
            for h in res.get("results", []):
                blocks.append(f"[{h['source']}] {h['text']}")
        else:
            blocks.append(f"[tool {e['tool']}({json.dumps(e['args'])})] {json.dumps(res)}")
    return "\n".join(blocks)


def judge_groundedness(answer: str, evidence: list[dict], judge: ChatModel | None = None) -> dict:
    answer = answer.replace(DISCLAIMER, "").strip()
    if not answer:
        return {"score": 0.0, "claims": []}
    judge = judge or get_chat_model(settings().judge_model)
    prompt = f"CONTEXT:\n{format_evidence(evidence) or '(none)'}\n\nANSWER:\n{answer}"
    out = judge.generate(JUDGE_SYSTEM, [types.Content(role="user", parts=[types.Part(text=prompt)])], json_mode=True)
    raw = "".join(p.text or "" for p in out.parts or [])
    try:
        claims = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))["claims"]
    except (AttributeError, json.JSONDecodeError, KeyError):
        return {"score": 0.0, "claims": [], "error": raw[:300]}
    if not claims:  # nothing factual to check (e.g. a refusal)
        return {"score": 1.0, "claims": []}
    score = sum(bool(c.get("supported")) for c in claims) / len(claims)
    return {"score": round(score, 4), "claims": claims}
