"""Deterministic financial tools shared by the MCP server, ADK agents and LangGraph."""
from advisor.tools.finance import (
    budget_50_30_20,
    compound_growth,
    debt_payoff_plan,
    emergency_fund_target,
    loan_amortization,
    retirement_projection,
)
from advisor.tools.market_data import get_price_history, portfolio_metrics


def search_knowledge_base(query: str, top_k: int = 4) -> dict:
    """Search the curated financial-education knowledge base (RAG).

    Use this for any conceptual question (what is X, how does Y work, pros/cons,
    rules of thumb) so the answer can be grounded in cited sources.

    Args:
        query: Natural-language search query.
        top_k: Number of passages to return (1-8).
    """
    from advisor.rag.retriever import get_retriever

    hits = get_retriever().search(query, top_k=max(1, min(int(top_k), 8)))
    return {"query": query, "results": [h.to_dict() for h in hits]}


TOOLS = {
    f.__name__: f
    for f in (
        search_knowledge_base,
        compound_growth,
        retirement_projection,
        loan_amortization,
        debt_payoff_plan,
        emergency_fund_target,
        budget_50_30_20,
        get_price_history,
        portfolio_metrics,
    )
}

__all__ = ["TOOLS", *TOOLS]
