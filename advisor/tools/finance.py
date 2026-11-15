"""Pure-Python personal-finance calculators.

Every function takes/returns JSON-serialisable values and has a Google-style
docstring, because the same functions are exposed to Gemini through ADK
FunctionTools, an MCP server and LangGraph.
"""
from __future__ import annotations

import math


def _r(x: float, n: int = 2) -> float:
    return round(float(x), n)


def compound_growth(
    principal: float,
    annual_rate_pct: float,
    years: int,
    monthly_contribution: float = 0.0,
) -> dict:
    """Project the future value of an investment with monthly compounding.

    Args:
        principal: Starting balance in dollars.
        annual_rate_pct: Expected nominal annual return, in percent (7 means 7%).
        years: Number of years to grow.
        monthly_contribution: Amount added at the end of every month.
    """
    if years < 0:
        raise ValueError("years must be >= 0")
    r = annual_rate_pct / 100 / 12
    n = int(round(years * 12))
    growth = (1 + r) ** n
    fv_principal = principal * growth
    fv_contrib = monthly_contribution * ((growth - 1) / r if r else n)
    fv = fv_principal + fv_contrib
    contributed = principal + monthly_contribution * n
    return {
        "future_value": _r(fv),
        "total_contributed": _r(contributed),
        "total_growth": _r(fv - contributed),
        "years": years,
        "annual_rate_pct": annual_rate_pct,
    }


def retirement_projection(
    current_age: int,
    retirement_age: int,
    current_savings: float,
    monthly_contribution: float,
    expected_return_pct: float = 6.0,
    annual_spending_in_retirement: float = 0.0,
    withdrawal_rate_pct: float = 4.0,
) -> dict:
    """Project savings at retirement and compare with the nest egg needed.

    The target nest egg is annual spending divided by the safe withdrawal rate
    (e.g. $40,000 / 4% = $1,000,000).

    Args:
        current_age: Age today.
        retirement_age: Planned retirement age.
        current_savings: Retirement savings today in dollars.
        monthly_contribution: Monthly amount saved toward retirement.
        expected_return_pct: Expected annual return in percent.
        annual_spending_in_retirement: Desired yearly spending in retirement (today's dollars).
        withdrawal_rate_pct: Sustainable withdrawal rate in percent.
    """
    years = retirement_age - current_age
    if years <= 0:
        raise ValueError("retirement_age must be greater than current_age")
    proj = compound_growth(current_savings, expected_return_pct, years, monthly_contribution)
    target = (
        annual_spending_in_retirement / (withdrawal_rate_pct / 100)
        if annual_spending_in_retirement
        else None
    )
    out = {
        "years_to_retirement": years,
        "projected_savings": proj["future_value"],
        "total_contributed": proj["total_contributed"],
        "target_nest_egg": _r(target) if target else None,
    }
    if target:
        gap = target - proj["future_value"]
        out["shortfall"] = _r(max(gap, 0))
        out["on_track"] = gap <= 0
        if gap > 0:
            r = expected_return_pct / 100 / 12
            n = years * 12
            factor = ((1 + r) ** n - 1) / r if r else n
            out["extra_monthly_needed"] = _r(gap / factor)
    return out


def loan_amortization(principal: float, annual_rate_pct: float, years: int) -> dict:
    """Compute the fixed monthly payment and total interest of an amortising loan.

    Args:
        principal: Amount borrowed in dollars.
        annual_rate_pct: Annual interest rate in percent (APR).
        years: Loan term in years.
    """
    n = int(years * 12)
    if n <= 0:
        raise ValueError("years must be > 0")
    r = annual_rate_pct / 100 / 12
    payment = principal / n if r == 0 else principal * r / (1 - (1 + r) ** -n)
    total = payment * n
    first_interest = principal * r
    return {
        "monthly_payment": _r(payment),
        "total_paid": _r(total),
        "total_interest": _r(total - principal),
        "first_payment_interest": _r(first_interest),
        "first_payment_principal": _r(payment - first_interest),
        "num_payments": n,
    }


def debt_payoff_plan(debts: list[dict], monthly_budget: float, strategy: str = "avalanche") -> dict:
    """Simulate paying off several debts with the avalanche or snowball method.

    Avalanche pays extra toward the highest APR first (least interest);
    snowball pays the smallest balance first (quick wins).

    Args:
        debts: List of {"name": str, "balance": float, "apr_pct": float, "min_payment": float}.
        monthly_budget: Total amount available for debt payments each month.
        strategy: "avalanche" or "snowball".
    """
    strategy = strategy.lower()
    if strategy not in {"avalanche", "snowball"}:
        raise ValueError("strategy must be 'avalanche' or 'snowball'")
    ds = [
        {
            "name": d.get("name", f"debt{i+1}"),
            "balance": float(d["balance"]),
            "apr": float(d["apr_pct"]) / 100 / 12,
            "min": float(d.get("min_payment", 0)),
        }
        for i, d in enumerate(debts)
    ]
    if monthly_budget < sum(d["min"] for d in ds):
        raise ValueError("monthly_budget is below the sum of minimum payments")
    key = (lambda d: -d["apr"]) if strategy == "avalanche" else (lambda d: d["balance"])
    order = [d["name"] for d in sorted(ds, key=key)]
    months, interest, payoff = 0, 0.0, {}
    while any(d["balance"] > 0.005 for d in ds) and months < 1200:
        months += 1
        for d in ds:
            if d["balance"] > 0:
                i = d["balance"] * d["apr"]
                interest += i
                d["balance"] += i
        budget = monthly_budget
        for d in ds:
            pay = min(d["min"], d["balance"])
            d["balance"] -= pay
            budget -= pay
        for name in order:
            d = next(x for x in ds if x["name"] == name)
            if budget <= 0:
                break
            pay = min(budget, d["balance"])
            d["balance"] -= pay
            budget -= pay
        for d in ds:
            if d["balance"] <= 0.005 and d["name"] not in payoff:
                payoff[d["name"]] = months
    return {
        "strategy": strategy,
        "payoff_order": order,
        "months_to_debt_free": months,
        "total_interest": _r(interest),
        "payoff_month_by_debt": payoff,
    }


def emergency_fund_target(monthly_expenses: float, months: int = 6, current_savings: float = 0.0) -> dict:
    """Compute an emergency-fund target (commonly 3-6 months of expenses) and the gap.

    Args:
        monthly_expenses: Essential monthly expenses in dollars.
        months: Months of expenses to cover.
        current_savings: Cash already set aside.
    """
    target = monthly_expenses * months
    gap = max(target - current_savings, 0)
    return {
        "target": _r(target),
        "current_savings": _r(current_savings),
        "gap": _r(gap),
        "months_covered": _r(current_savings / monthly_expenses if monthly_expenses else math.inf, 1),
    }


def budget_50_30_20(monthly_net_income: float) -> dict:
    """Split take-home pay using the 50/30/20 rule (needs / wants / savings & debt).

    Args:
        monthly_net_income: Monthly after-tax income in dollars.
    """
    return {
        "needs_50pct": _r(monthly_net_income * 0.5),
        "wants_30pct": _r(monthly_net_income * 0.3),
        "savings_and_debt_20pct": _r(monthly_net_income * 0.2),
    }
