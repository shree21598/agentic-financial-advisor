# Agentic Financial Advisor

**Vertex AI · Gemini · Google ADK · LangGraph · MCP · RAG**

A multi-agent financial advisor that answers personal-finance questions with **grounded, cited** explanations and **tool-computed** numbers. It ships with an evaluation harness that measures **groundedness**, **tool-call accuracy** and **p95 latency** over a 100-query test set.

```
                         ┌──────────────────────── MCP server (FastMCP, stdio / HTTP) ───────────────────────┐
                         │ search_knowledge_base · compound_growth · retirement_projection · loan_amortization│
                         │ debt_payoff_plan · emergency_fund_target · budget_50_30_20                        │
                         │ get_price_history · portfolio_metrics                                             │
                         └───────────────▲──────────────────────────────────────────▲────────────────────────┘
                                         │ McpToolset                                │ MCP ClientSession
  ┌───────────── Google ADK (deploys to Agent Engine) ─────────────┐   ┌──────────── LangGraph ────────────────┐
  │ financial_advisor (Gemini coordinator, input guardrail)        │   │ guardrail → agent ⇄ tools             │
  │   ├─ AgentTool(research_agent)  → RAG with citations           │   │        → grounding_check (LLM judge,  │
  │   ├─ AgentTool(planning_agent)  → calculators                  │   │          revise loop) → finalize      │
  │   └─ AgentTool(market_agent)    → price / portfolio analytics  │   │          (compliance disclaimer)      │
  └────────────────────────────────────────────────────────────────┘   └───────────────────────────────────────┘
                                         │
               RAG: markdown corpus → heading-aware chunks → Vertex AI embeddings (text-embedding-005)
                    + BM25 → reciprocal-rank fusion → top-k passages with source citations
```

## What's inside

| Path | Purpose |
|---|---|
| `advisor/tools/` | Deterministic calculators + market analytics (the model never does math itself) |
| `advisor/rag/` | Knowledge corpus (13 docs), chunker, Vertex embeddings, hybrid dense+BM25 retriever |
| `advisor/mcp_server/` | MCP server exposing every tool (stdio or streamable HTTP) |
| `advisor/adk_agent/` | ADK coordinator + 3 specialist agents; tools in-process or via `McpToolset` |
| `advisor/graph/` | LangGraph workflow, LLM-as-judge groundedness check, guardrails/compliance |
| `eval/` | 100-query dataset generator, metrics, runner, markdown report |
| `scripts/deploy_agent_engine.py` | Deploy the ADK agent to Vertex AI Agent Engine |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set GOOGLE_CLOUD_PROJECT etc.
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com
```

Ask a question through the LangGraph agent:

```bash
python -m advisor.graph.workflow "I'm 35 with \$80k saved, saving \$1,500/month. Can I retire at 65 on \$60k a year?"
TOOL_TRANSPORT=mcp python -m advisor.graph.workflow "Volatility and Sharpe of 60% VTI / 40% BND?"
```

Run the ADK multi-agent version in the ADK dev UI:

```bash
adk web advisor           # pick "adk_agent"
ADK_USE_MCP=1 adk web advisor   # tools served by the MCP server instead of in-process
```

Run the MCP server for any MCP client (Claude Desktop, Cursor, ADK, ...):

```bash
python -m advisor.mcp_server.server            # stdio
python -m advisor.mcp_server.server --http 8080
```

No GCP account? Everything runs offline with a deterministic stub model and hashing embeddings:

```bash
ADVISOR_OFFLINE=1 pytest -q
ADVISOR_OFFLINE=1 python -m eval.run_eval
```

## Evaluation

```bash
python eval/build_dataset.py                       # regenerates eval/dataset.jsonl (seeded)
python -m eval.run_eval                            # LangGraph + Gemini on Vertex AI
python -m eval.run_eval --engine adk               # ADK multi-agent
python -m eval.run_eval --inline-grounding         # with the self-check / revise loop
```

**Dataset (100 queries):** 30 conceptual (RAG), 10 compound growth, 10 retirement, 10 loans, 8 debt payoff, 6 emergency fund, 6 budgeting, 6 single-ticker, 8 portfolio, 6 multi-tool.

**Metrics**
- **Groundedness**: a Gemini judge (`JUDGE_MODEL`) splits each answer into atomic claims and checks each against the evidence the agent actually retrieved (RAG passages + tool outputs); score = supported / total, averaged over queries.
- **Tool-call accuracy**: a query counts as correct only if every expected tool was called successfully, no unexpected tool was called (an extra `search_knowledge_base` is tolerated), **and** key arguments match the question within 1%.
- **Latency**: end-to-end wall time per query (concurrency 1), reported as p50 / p95.

Results are written to `eval/results/<engine>_<mode>_report.md` (+ full JSON traces).

### Results

Run `python -m eval.run_eval` against your Vertex AI project and paste the summary table from
`eval/results/graph_vertex_report.md` here.

| Engine | Model | Groundedness | Tool-call accuracy | p95 latency |
|---|---|---|---|---|
| LangGraph | gemini-2.5-flash | _run eval_ | _run eval_ | _run eval_ |
| ADK | gemini-2.5-flash | _run eval_ | _run eval_ | _run eval_ |

## Deploy to Vertex AI Agent Engine

```bash
pip install "google-cloud-aiplatform[agent_engines,adk]"
export STAGING_BUCKET=gs://your-bucket
python scripts/deploy_agent_engine.py --test
```

## Design notes

- **Numbers come from tools, words come from sources.** The system prompt forbids model arithmetic; every figure is produced by a tested calculator, and every concept is retrieved and cited. This is what drives groundedness.
- **Hybrid retrieval.** Dense embeddings miss exact terms ("wash-sale", "Rule of 72"); BM25 misses paraphrases. RRF fusion gets both without score calibration.
- **One toolset, three surfaces.** The same Python functions are exposed as ADK function tools, MCP tools and LangGraph tool calls, so evaluations compare orchestration, not tool implementations.
- **Guardrails.** Clearly illegal requests (insider trading, tax evasion, laundering) are refused before any model call; every answer gets a compliance disclaimer.
- **Market data is synthetic.** `data/sample_prices_synthetic.csv` is generated (`scripts/generate_sample_prices.py`) so evals are reproducible offline. Set `PRICES_CSV` to a real `date,ticker,close` file for real data.

> Educational project - not financial advice.
