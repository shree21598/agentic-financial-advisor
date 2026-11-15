"""Generate the SYNTHETIC daily price file used for offline, reproducible evals.

Correlated geometric Brownian motion; parameters are rough, illustrative asset-class
assumptions, NOT real market history.
"""
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ASSETS = {  # ticker: (annual drift, annual vol, start price, factor loading on equity)
    "VTI": (0.08, 0.17, 200.0, 1.0),
    "VXUS": (0.06, 0.18, 55.0, 0.85),
    "QQQ": (0.10, 0.23, 350.0, 1.1),
    "BND": (0.03, 0.06, 75.0, -0.1),
    "GLD": (0.04, 0.15, 170.0, 0.05),
    "VNQ": (0.06, 0.21, 85.0, 0.7),
}
DAYS = 252 * 5


def main(out: Path = Path(__file__).resolve().parent.parent / "data" / "sample_prices_synthetic.csv"):
    rng = np.random.default_rng(42)
    market = rng.standard_normal(DAYS)
    d0 = date(2021, 1, 4)
    dates, d = [], d0
    while len(dates) < DAYS:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += timedelta(days=1)
    rows = []
    for t, (mu, vol, p0, beta) in ASSETS.items():
        idio = rng.standard_normal(DAYS)
        rho = max(min(beta, 0.95), -0.95)
        z = rho * market + np.sqrt(1 - rho**2) * idio
        dt = 1 / 252
        logret = (mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
        prices = p0 * np.exp(np.cumsum(logret))
        rows += [(dt_, t, round(float(p), 4)) for dt_, p in zip(dates, prices)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "close"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
