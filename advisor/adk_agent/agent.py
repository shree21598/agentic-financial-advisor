"""Google ADK multi-agent financial advisor.

    financial_advisor (coordinator, Gemini)
      ├── AgentTool(research_agent)   -> search_knowledge_base           (RAG)
      ├── AgentTool(planning_agent)   -> compound/retirement/loan/debt/budget calculators
      └── AgentTool(market_agent)     -> get_price_history, portfolio_metrics

Tools are plain Python functions by default, or are served by the MCP server when
ADK_USE_MCP=1 (McpToolset over stdio). Run locally with `adk web advisor` or
`adk run advisor/adk_agent`; deploy with `python scripts/deploy_agent_engine.py`.
"""
from __future__ import annotations

import os
import sys

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from advisor import tools as T
from advisor.config import settings
from advisor.graph.compliance import DISCLAIMER, check_input
from advisor.prompts import ADVISOR_SYSTEM

MODEL = settings().model

RESEARCH_TOOLS = ["search_knowledge_base"]
PLANNING_TOOLS = [
    "compound_growth", "retirement_projection", "loan_amortization",
    "debt_payoff_plan", "emergency_fund_target", "budget_50_30_20",
]
MARKET_TOOLS = ["get_price_history", "portfolio_metrics"]

# --------------------------------------------------------------------- tool-call recorder (for evals)
TOOL_LOG: list[dict] = []


def record_tool(tool, args, tool_context, tool_response=None):
    TOOL_LOG.append({"tool": tool.name, "args": dict(args), "result": tool_response})
    return None


def _tools(names: list[str]):
    if os.getenv("ADK_USE_MCP", "0") == "1":
        from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
        from mcp import StdioServerParameters

        return [
            McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=sys.executable, args=["-m", "advisor.mcp_server.server"], env=dict(os.environ)
                    ),
                    timeout=30,
                ),
                tool_filter=names,
            )
        ]
    return [T.TOOLS[n] for n in names]


def _specialist(name: str, description: str, instruction: str, tool_names: list[str]) -> LlmAgent:
    return LlmAgent(
        name=name,
        model=MODEL,
        description=description,
        instruction=instruction,
        tools=_tools(tool_names),
        after_tool_callback=record_tool,
    )


research_agent = _specialist(
    "research_agent",
    "Answers conceptual personal-finance questions using the curated knowledge base, with citations.",
    "Call search_knowledge_base (rephrase and search again if results are off-topic). Answer ONLY from the "
    "returned passages and cite each fact as [source]. If the passages do not cover it, say so.",
    RESEARCH_TOOLS,
)
planning_agent = _specialist(
    "planning_agent",
    "Runs financial calculations: compound growth, retirement projections, loans, debt payoff, emergency fund, budgets.",
    "Pick the right calculator, fill arguments from the request (state any assumed values), call it, and "
    "report the returned numbers exactly. Never compute numbers yourself.",
    PLANNING_TOOLS,
)
market_agent = _specialist(
    "market_agent",
    "Analyses tickers and portfolios: price history, return, volatility, Sharpe ratio, drawdown, correlation.",
    "Use get_price_history for one ticker and portfolio_metrics for a weighted mix. Report numbers exactly and "
    "explain what they mean in one sentence each. Mention that the bundled data is synthetic if data_note says so.",
    MARKET_TOOLS,
)


def input_guardrail(callback_context, llm_request):
    """before_model_callback: short-circuit clearly illegal requests without calling Gemini."""
    from google.adk.models import LlmResponse
    from google.genai import types

    last = next(
        (p.text for c in reversed(llm_request.contents or []) if c.role == "user" for p in c.parts or [] if p.text),
        "",
    )
    refusal = check_input(last)
    if refusal:
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=f"{refusal}\n\n{DISCLAIMER}")]))
    return None


root_agent = LlmAgent(
    name="financial_advisor",
    model=MODEL,
    description="Multi-agent educational financial advisor.",
    instruction=ADVISOR_SYSTEM
    + "\nYou coordinate three specialists exposed as tools: research_agent (concepts, with citations), "
    "planning_agent (all calculations) and market_agent (tickers/portfolios). Delegate every factual or "
    "numeric sub-question to the right specialist (in parallel when independent), then synthesise one answer "
    "that keeps their citations and numbers verbatim. End with this disclaimer exactly:\n" + DISCLAIMER,
    tools=[AgentTool(research_agent), AgentTool(planning_agent), AgentTool(market_agent)],
    before_model_callback=input_guardrail,
)
