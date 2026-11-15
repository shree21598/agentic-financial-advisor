"""Generate the 100-query evaluation set (eval/dataset.jsonl), reproducibly (seed=7).

Each row: id, category, query, expected_tools, expected_args (key numeric args that a
correct call must contain, compared with 1% tolerance).
"""
import json
import random
from pathlib import Path

rng = random.Random(7)
OUT = Path(__file__).with_name("dataset.jsonl")

CONCEPTS = [
    "What is the 4% rule in retirement planning?",
    "How big should my emergency fund be?",
    "Should I pay off credit card debt or invest first?",
    "What is the difference between the debt avalanche and debt snowball methods?",
    "Why do expense ratios matter so much for index funds?",
    "What is the Rule of 72?",
    "How should my asset allocation change as I get older?",
    "What is diversification and why does it reduce risk?",
    "Roth vs traditional 401(k): which is better for a young worker?",
    "What happens if I withdraw from my IRA before 59 1/2?",
    "What does the Sharpe ratio measure?",
    "What is maximum drawdown?",
    "Why do bond prices fall when interest rates rise?",
    "What is bond duration?",
    "Is dollar-cost averaging better than investing a lump sum?",
    "How often should I rebalance my portfolio?",
    "What is tax-loss harvesting and what is the wash-sale rule?",
    "What is asset location?",
    "Where should I keep my emergency fund?",
    "What is the 50/30/20 budgeting rule?",
    "ETFs vs mutual funds - what's the difference?",
    "Should I get term or whole life insurance?",
    "Why is an employer 401(k) match so valuable?",
    "What are the risks of a balance transfer card?",
    "What's the difference between nominal and real returns?",
    "What is the difference between short-term and long-term capital gains?",
    "What are the benefits of an HSA?",
    "What does correlation mean for a portfolio?",
    "Do actively managed funds beat index funds?",
    "Why is the 4% rule risky for early retirees?",
]


def money(lo, hi, step):
    return rng.randrange(lo, hi + 1, step)


def rows():
    out = []

    def add(cat, q, tools, args=None):
        out.append({"category": cat, "query": q, "expected_tools": tools, "expected_args": args or {}})

    for q in CONCEPTS:
        add("concept", q, ["search_knowledge_base"])
    for _ in range(10):
        p, m, r, y = money(1000, 50000, 1000), money(100, 2000, 50), rng.choice([5, 6, 7, 8]), rng.choice([10, 15, 20, 25, 30])
        add("compound_growth", f"If I invest ${p:,} now and add ${m:,} per month at {r}% a year, how much will I have in {y} years?",
            ["compound_growth"], {"compound_growth": {"principal": p, "monthly_contribution": m, "annual_rate_pct": r, "years": y}})
    for _ in range(10):
        a, ra = rng.randint(25, 45), rng.choice([60, 62, 65, 67])
        s, m, sp = money(5000, 200000, 5000), money(200, 2500, 100), money(40000, 100000, 5000)
        add("retirement", f"I'm {a}, want to retire at {ra}, have ${s:,} saved and save ${m:,} a month. Assuming 6% returns, can I spend ${sp:,} a year in retirement?",
            ["retirement_projection"], {"retirement_projection": {"current_age": a, "retirement_age": ra, "current_savings": s, "monthly_contribution": m, "annual_spending_in_retirement": sp}})
    for _ in range(10):
        p, r, y = money(100000, 800000, 10000), rng.choice([5.5, 6.0, 6.5, 7.0, 7.25]), rng.choice([15, 30])
        add("loan", f"What is the monthly payment and total interest on a ${p:,} mortgage at {r}% for {y} years?",
            ["loan_amortization"], {"loan_amortization": {"principal": p, "annual_rate_pct": r, "years": y}})
    for i in range(8):
        strat = "avalanche" if i % 2 == 0 else "snowball"
        cb, ca = money(1000, 9000, 500), rng.choice([19.9, 22.0, 24.5, 27.0])
        lb, la = money(5000, 20000, 1000), rng.choice([5.0, 6.5, 8.0])
        budget = money(700, 1500, 50)
        add("debt", f"Make a {strat} payoff plan: card ${cb:,} at {ca}% (min $100) and loan ${lb:,} at {la}% (min $250), with ${budget:,} per month.",
            ["debt_payoff_plan"], {"debt_payoff_plan": {"monthly_budget": budget, "strategy": strat}})
    for _ in range(6):
        e, s, mo = money(2000, 7000, 250), money(0, 10000, 500), rng.choice([3, 6])
        add("emergency", f"My essential expenses are ${e:,} a month and I have ${s:,} saved. How far am I from a {mo}-month emergency fund?",
            ["emergency_fund_target"], {"emergency_fund_target": {"monthly_expenses": e, "current_savings": s, "months": mo}})
    for _ in range(6):
        inc = money(2500, 12000, 250)
        add("budget", f"I take home ${inc:,} per month. How should I split it using a 50/30/20 budget?",
            ["budget_50_30_20"], {"budget_50_30_20": {"monthly_net_income": inc}})
    for t in ["VTI", "QQQ", "BND", "GLD", "VXUS", "VNQ"]:
        add("price_history", f"How has {t} performed over the last year, and how volatile was it?",
            ["get_price_history"], {"get_price_history": {"ticker": t}})
    mixes = [("VTI", 60, "BND", 40), ("VTI", 80, "BND", 20), ("QQQ", 50, "BND", 50), ("VTI", 70, "GLD", 30),
             ("VTI", 50, "VXUS", 50), ("VNQ", 30, "BND", 70), ("QQQ", 60, "GLD", 40), ("VXUS", 40, "BND", 60)]
    for a, wa, b, wb in mixes:
        add("portfolio", f"What are the return, volatility, Sharpe ratio and max drawdown of a portfolio of {wa}% {a} and {wb}% {b}?",
            ["portfolio_metrics"], {"portfolio_metrics": {}})
    add("multi", "Explain the 4% rule, then tell me if I'm on track: I'm 35, retiring at 65 with $80,000 saved, saving $1,500 a month, and want $60,000 a year.",
        ["search_knowledge_base", "retirement_projection"], {"retirement_projection": {"current_age": 35, "retirement_age": 65, "current_savings": 80000, "monthly_contribution": 1500, "annual_spending_in_retirement": 60000}})
    add("multi", "Why does avalanche save interest? Build the avalanche plan for card $6,000 at 24% (min $120) and car $15,000 at 7% (min $300) with $1,000 per month.",
        ["search_knowledge_base", "debt_payoff_plan"], {"debt_payoff_plan": {"monthly_budget": 1000, "strategy": "avalanche"}})
    add("multi", "How big should an emergency fund be and where should I keep it? My expenses are $3,500 a month and I have $4,000 saved.",
        ["search_knowledge_base", "emergency_fund_target"], {"emergency_fund_target": {"monthly_expenses": 3500, "current_savings": 4000}})
    add("multi", "What does diversification do, and what is the correlation and volatility of 60% VTI / 40% BND?",
        ["search_knowledge_base", "portfolio_metrics"], {"portfolio_metrics": {}})
    add("multi", "I take home $6,000 a month. Split it with 50/30/20 and tell me how much $1,200 a month invested at 7% grows to in 25 years.",
        ["budget_50_30_20", "compound_growth"], {"budget_50_30_20": {"monthly_net_income": 6000}, "compound_growth": {"monthly_contribution": 1200, "annual_rate_pct": 7, "years": 25}})
    add("multi", "Compare QQQ's last-year performance to a 50/50 VTI and BND portfolio.",
        ["get_price_history", "portfolio_metrics"], {"get_price_history": {"ticker": "QQQ"}})
    for i, r in enumerate(out, 1):
        r["id"] = f"q{i:03d}"
    return out


if __name__ == "__main__":
    data = rows()
    assert len(data) == 100, len(data)
    OUT.write_text("\n".join(json.dumps(r) for r in data) + "\n")
    print(f"wrote {len(data)} queries -> {OUT}")
