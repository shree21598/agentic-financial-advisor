"""Evaluation metrics: tool-call accuracy, groundedness and latency percentiles."""
from __future__ import annotations

import math

OPTIONAL_TOOLS = {"search_knowledge_base"}  # extra grounding lookups are not penalised on calc queries


def _close(a, b, tol=0.01) -> bool:
    if isinstance(b, str):
        return str(a).strip().lower() == b.lower()
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isclose(a, b, rel_tol=tol, abs_tol=1e-6)


def tool_call_score(expected_tools: list[str], expected_args: dict, calls: list[dict]) -> dict:
    """Score one query.

    selection_correct: every expected tool was called successfully and no unexpected tool was
                       called (search_knowledge_base is tolerated as an extra).
    args_correct:      for each expected tool, at least one successful call matches every key
                       argument in expected_args within 1%.
    correct:           selection_correct and args_correct  (the headline "tool call accuracy").
    """
    ok_calls = [c for c in calls if c.get("ok", True)]
    called = {c["tool"] for c in ok_calls}
    exp = set(expected_tools)
    extras = {c["tool"] for c in calls} - exp - OPTIONAL_TOOLS
    selection = exp <= called and not extras
    args_ok = True
    for tool, want in (expected_args or {}).items():
        cands = [c["args"] for c in ok_calls if c["tool"] == tool]
        if not any(all(k in a and _close(a[k], v) for k, v in want.items()) for a in cands):
            args_ok = False
    tp = len(exp & called)
    return {
        "selection_correct": selection,
        "args_correct": args_ok,
        "correct": selection and args_ok,
        "precision": tp / len({c["tool"] for c in calls}) if calls else (1.0 if not exp else 0.0),
        "recall": tp / len(exp) if exp else 1.0,
        "unexpected_tools": sorted(extras),
        "failed_calls": [c["tool"] for c in calls if not c.get("ok", True)],
    }


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[dict]) -> dict:
    n = len(results)
    lat = [r["latency_s"] for r in results]
    g = [r["groundedness"]["score"] for r in results if r.get("groundedness")]
    by_cat: dict[str, dict] = {}
    for r in results:
        c = by_cat.setdefault(r["category"], {"n": 0, "tool_correct": 0, "groundedness": []})
        c["n"] += 1
        c["tool_correct"] += r["tool_score"]["correct"]
        if r.get("groundedness"):
            c["groundedness"].append(r["groundedness"]["score"])
    return {
        "n": n,
        "groundedness_mean": round(sum(g) / len(g), 4) if g else None,
        "tool_call_accuracy": round(sum(r["tool_score"]["correct"] for r in results) / n, 4),
        "tool_selection_accuracy": round(sum(r["tool_score"]["selection_correct"] for r in results) / n, 4),
        "tool_recall_mean": round(sum(r["tool_score"]["recall"] for r in results) / n, 4),
        "latency_p50_s": round(percentile(lat, 50), 3),
        "latency_p95_s": round(percentile(lat, 95), 3),
        "latency_mean_s": round(sum(lat) / n, 3),
        "errors": sum(1 for r in results if r.get("error")),
        "by_category": {
            k: {
                "n": v["n"],
                "tool_call_accuracy": round(v["tool_correct"] / v["n"], 3),
                "groundedness": round(sum(v["groundedness"]) / len(v["groundedness"]), 3) if v["groundedness"] else None,
            }
            for k, v in sorted(by_cat.items())
        },
    }
