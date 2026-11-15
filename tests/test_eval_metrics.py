from eval.metrics import percentile, tool_call_score


def test_tool_score_exact():
    s = tool_call_score(["loan_amortization"], {"loan_amortization": {"principal": 100000}},
                        [{"tool": "loan_amortization", "args": {"principal": 100000.0}, "ok": True}])
    assert s["correct"]


def test_tool_score_wrong_args_and_extras():
    calls = [{"tool": "loan_amortization", "args": {"principal": 90000}, "ok": True}]
    assert not tool_call_score(["loan_amortization"], {"loan_amortization": {"principal": 100000}}, calls)["correct"]
    calls.append({"tool": "compound_growth", "args": {}, "ok": True})
    assert not tool_call_score(["loan_amortization"], {}, calls)["selection_correct"]


def test_search_is_tolerated_extra():
    calls = [{"tool": "budget_50_30_20", "args": {"monthly_net_income": 5000}, "ok": True},
             {"tool": "search_knowledge_base", "args": {"query": "x"}, "ok": True}]
    assert tool_call_score(["budget_50_30_20"], {}, calls)["correct"]


def test_percentile():
    assert percentile(list(range(1, 101)), 95) == 95.05
