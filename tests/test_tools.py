import pytest

from advisor.tools import TOOLS
from advisor.tools.finance import (
    budget_50_30_20, compound_growth, debt_payoff_plan, emergency_fund_target, loan_amortization, retirement_projection,
)
from advisor.tools.market_data import get_price_history, portfolio_metrics
from advisor.tools.schema import function_schema


def test_loan_matches_standard_formula():
    # $300k, 6.5%, 30y -> $1,896.20 (standard amortisation tables)
    assert loan_amortization(300_000, 6.5, 30)["monthly_payment"] == pytest.approx(1896.20, abs=0.01)


def test_zero_rate_loan():
    assert loan_amortization(12_000, 0, 1)["monthly_payment"] == 1000


def test_compound_growth_doubling_rule_of_72():
    fv = compound_growth(10_000, 7.2, 10)["future_value"]
    assert 19_500 < fv < 21_000


def test_retirement_target_uses_withdrawal_rate():
    r = retirement_projection(30, 65, 0, 0, 6, 40_000)
    assert r["target_nest_egg"] == 1_000_000 and r["on_track"] is False


def test_avalanche_beats_snowball_on_interest():
    debts = [
        {"name": "small", "balance": 1000, "apr_pct": 5, "min_payment": 50},
        {"name": "big", "balance": 8000, "apr_pct": 25, "min_payment": 150},
    ]
    a = debt_payoff_plan(debts, 600, "avalanche")
    s = debt_payoff_plan(debts, 600, "snowball")
    assert a["payoff_order"] == ["big", "small"] and s["payoff_order"] == ["small", "big"]
    assert a["total_interest"] < s["total_interest"]


def test_debt_budget_below_minimums_raises():
    with pytest.raises(ValueError):
        debt_payoff_plan([{"name": "x", "balance": 100, "apr_pct": 10, "min_payment": 50}], 10)


def test_emergency_and_budget():
    assert emergency_fund_target(3000, 6, 4000)["gap"] == 14000
    assert budget_50_30_20(5000) == {"needs_50pct": 2500, "wants_30pct": 1500, "savings_and_debt_20pct": 1000}


def test_market_tools():
    h = get_price_history("vti")
    assert h["ticker"] == "VTI" and h["annualized_volatility_pct"] > 0
    p = portfolio_metrics({"VTI": 3, "BND": 2})
    assert p["weights"] == {"VTI": 0.6, "BND": 0.4}
    assert p["correlation"]["VTI"]["VTI"] == 1.0
    with pytest.raises(ValueError):
        get_price_history("NOPE")


@pytest.mark.parametrize("name", list(TOOLS))
def test_every_tool_has_schema_with_described_params(name):
    s = function_schema(TOOLS[name])
    assert s["description"]
    for p, spec in s["parameters"]["properties"].items():
        assert spec.get("description"), f"{name}.{p} missing description"
