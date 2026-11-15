"""Run the advisor over eval/dataset.jsonl and report groundedness, tool-call accuracy and latency.

    python -m eval.run_eval                      # LangGraph engine, Gemini on Vertex AI
    python -m eval.run_eval --engine adk         # Google ADK multi-agent engine
    python -m eval.run_eval --transport mcp      # route tool calls through the MCP server
    ADVISOR_OFFLINE=1 python -m eval.run_eval    # offline smoke test (stub model; not real metrics)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from advisor.config import settings
from advisor.graph.grounding import judge_groundedness
from eval.metrics import summarize, tool_call_score

HERE = Path(__file__).parent


async def run(args) -> dict:
    data = [json.loads(l) for l in (HERE / "dataset.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        data = data[: args.limit]
    sem = asyncio.Semaphore(args.concurrency)

    if args.engine == "adk":
        from advisor.adk_agent.runner import ADKAdvisor

        adk = ADKAdvisor()
        if args.concurrency > 1:
            raise SystemExit("--engine adk records tool calls globally; use --concurrency 1")

        async def answer(q):
            return await adk.ask(q)

        ctx = None
    else:
        from advisor.graph.workflow import build_graph
        from advisor.tools.executor import get_executor

        ctx = get_executor(args.transport)
        ex = await ctx.__aenter__()
        graph = build_graph(ex, inline_grounding=args.inline_grounding)

        async def answer(q):
            st = await graph.ainvoke({"query": q}, {"recursion_limit": 30})
            return {"answer": st.get("answer", ""), "tool_calls": st.get("tool_calls", []), "evidence": st.get("evidence", [])}

    async def one(row):
        async with sem:
            t0 = time.perf_counter()
            try:
                out, err = await answer(row["query"]), None
            except Exception as e:  # keep going; count as failure
                out, err = {"answer": "", "tool_calls": [], "evidence": []}, repr(e)
            latency = time.perf_counter() - t0
            g = await asyncio.to_thread(judge_groundedness, out["answer"], out["evidence"]) if out["answer"] else None
            res = {
                **row,
                "answer": out["answer"],
                "tool_calls": out["tool_calls"],
                "tool_score": tool_call_score(row["expected_tools"], row["expected_args"], out["tool_calls"]),
                "groundedness": g,
                "latency_s": round(latency, 3),
                "error": err,
            }
            mark = "ok " if res["tool_score"]["correct"] else "BAD"
            print(f"{row['id']} {mark} g={g['score'] if g else '-':<5} {latency:5.2f}s  {row['query'][:70]}", flush=True)
            return res

    try:
        results = await asyncio.gather(*(one(r) for r in data))
    finally:
        if ctx is not None:
            await ctx.__aexit__(None, None, None)

    s = settings()
    summary = summarize(results)
    summary["config"] = {
        "engine": args.engine,
        "transport": args.transport,
        "model": "offline-stub" if s.offline else s.model,
        "judge_model": "offline-lexical" if s.offline else s.judge_model,
        "embedding_model": "offline-hashing" if s.offline else s.embedding_model,
        "inline_grounding": args.inline_grounding,
        "concurrency": args.concurrency,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return {"summary": summary, "results": results}


def write_report(out: dict, path: Path) -> None:
    s = out["summary"]
    c = s["config"]
    lines = [
        f"# Evaluation report ({c['engine']}, {c['model']})",
        "",
        f"_{c['timestamp']} - {s['n']} queries - judge: {c['judge_model']} - transport: {c['transport']} - concurrency: {c['concurrency']}_",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Groundedness (mean claim support) | {s['groundedness_mean']:.1%} |" if s["groundedness_mean"] is not None else "| Groundedness | n/a |",
        f"| Tool-call accuracy (right tools + right args) | {s['tool_call_accuracy']:.1%} |",
        f"| Tool-selection accuracy | {s['tool_selection_accuracy']:.1%} |",
        f"| Latency p50 / p95 | {s['latency_p50_s']:.2f}s / {s['latency_p95_s']:.2f}s |",
        f"| Errors | {s['errors']} |",
        "", "## By category", "", "| Category | n | Tool-call acc. | Groundedness |", "|---|---|---|---|",
    ]
    for k, v in s["by_category"].items():
        g = f"{v['groundedness']:.1%}" if v["groundedness"] is not None else "-"
        lines.append(f"| {k} | {v['n']} | {v['tool_call_accuracy']:.1%} | {g} |")
    fails = [r for r in out["results"] if not r["tool_score"]["correct"]]
    if fails:
        lines += ["", "## Tool-call failures", ""]
        for r in fails[:25]:
            got = [t["tool"] for t in r["tool_calls"]]
            lines.append(f"- `{r['id']}` expected {r['expected_tools']} got {got} - {r['query'][:90]}")
    if "offline" in c["model"]:
        lines += ["", "> Offline stub run: these numbers validate the pipeline only and are NOT model-quality metrics."]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["graph", "adk"], default="graph")
    ap.add_argument("--transport", choices=["local", "mcp"], default="local")
    ap.add_argument("--concurrency", type=int, default=1, help="keep 1 for faithful latency numbers")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--inline-grounding", action="store_true", help="enable the self-check/revise loop")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    out = asyncio.run(run(args))
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    tag = f"{args.engine}_{'offline' if settings().offline else 'vertex'}"
    (od / f"{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    write_report(out, od / f"{tag}_report.md")
    print(json.dumps({k: v for k, v in out["summary"].items() if k != "by_category"}, indent=1))
    print(f"report -> {od / f'{tag}_report.md'}")


if __name__ == "__main__":
    main()
