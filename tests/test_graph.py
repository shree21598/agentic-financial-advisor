from advisor.graph.compliance import DISCLAIMER
from advisor.graph.workflow import ask
from advisor.tools.executor import LocalToolExecutor, MCPToolExecutor


async def test_graph_calls_calculator_and_adds_disclaimer():
    st = await ask("What is the monthly payment on a $200,000 mortgage at 6% for 30 years?", LocalToolExecutor())
    assert [c["tool"] for c in st["tool_calls"]][0] == "loan_amortization"
    assert "1199.1" in st["answer"] and DISCLAIMER in st["answer"]


async def test_graph_refuses_illegal_request():
    st = await ask("Give me insider trading tips", LocalToolExecutor())
    assert st["blocked"] and not st.get("tool_calls")


async def test_inline_grounding_loop_runs():
    st = await ask("What is the Rule of 72?", LocalToolExecutor(), inline_grounding=True)
    assert "groundedness" in st and 0 <= st["groundedness"]["score"] <= 1


async def test_mcp_server_roundtrip():
    async with MCPToolExecutor() as ex:
        names = {d["name"] for d in await ex.declarations()}
        assert {"search_knowledge_base", "portfolio_metrics", "loan_amortization"} <= names
        res = await ex.call("budget_50_30_20", {"monthly_net_income": 4000})
        assert res["needs_50pct"] == 2000
