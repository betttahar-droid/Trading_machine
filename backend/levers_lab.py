"""
Long and short trend books with separate levers, and the 100-a-month deposit plan.

1. Long/short. The live trend rules on the 8 coins (backtest_trend.simulate: next-open fills, resting
   stops, fees, slippage, real funding), run as a long book, a short book, and combined with a different
   risk per side (TrendParams.short_risk_pct). In-sample 2020-06 .. 2024-06, out-of-sample 2024-07 .. now.

2. Deposit plan. 500 to start plus 100 every month, target 10,000:
   - the time it takes at a fixed yearly return, and
   - the historical odds for each book and risk level: a fresh account every 2 weeks since 2020-06,
     deposits added monthly, L x the book's daily returns (L = risk per trade / 1%).

    python -m backend.levers_lab
"""

import time
from dataclasses import replace
from typing import Dict

import numpy as np
import pandas as pd

from backend.backtest_trend import SPLIT, UNIVERSE, _ms, simulate
from backend.growth_study import LIVE, load_market_bulk
from backend.strategy_lab import START, fmt
from backend.universe_data import available_months

START_EQ, MONTHLY, TARGET = 500.0, 100.0, 10_000.0


def months_to_target(yearly: float, start: float = START_EQ, monthly: float = MONTHLY, target: float = TARGET) -> int:
    eq, m = start, 0
    g = (1 + yearly) ** (1 / 12)
    while eq < target and m < 600:
        eq = eq * g + monthly
        m += 1
    return m


def deposit_odds(r: pd.Series, L: float, step: int = 14) -> dict:
    """Fresh 500 account every `step` days, +100 every 30 days, daily equity x (1 + L r)."""
    x = r.to_numpy()
    res = {12: [], 24: [], 36: []}
    end24 = []
    for s in range(0, len(x), step):
        eq, hit, day = START_EQ, None, 0
        for t in range(s, min(len(x), s + 36 * 30)):
            eq = max(eq * (1 + L * x[t]), 0.0)
            day += 1
            if day % 30 == 0:
                eq += MONTHLY
            if hit is None and eq >= TARGET:
                hit = day
            if day == 24 * 30:
                end24.append(eq / (START_EQ + 24 * MONTHLY))
        for h in res:
            if s + h * 30 <= len(x):           # only windows with the full horizon observed
                res[h].append(hit is not None and hit <= h * 30)
    out = {f"10k<={h}mo": (np.mean(v) if v else np.nan) for h, v in res.items()}
    e = np.array(end24)
    out["median 24mo / deposited"] = float(np.median(e)) if len(e) else np.nan
    out["worst 24mo / deposited"] = float(e.min()) if len(e) else np.nan
    return out


def bootstrap_odds(r: pd.Series, L: float, n_paths: int = 2000, block: int = 30, seed: int = 0) -> dict:
    """36-month paths stitched from random 30-day blocks of r (keeps volatility clustering), +100/month."""
    rng = np.random.default_rng(seed)
    x = r.to_numpy()
    hits24 = hits36 = below = 0
    months = []
    for _ in range(n_paths):
        starts = rng.integers(0, len(x) - block, size=36)
        path = np.concatenate([x[s:s + block] for s in starts])
        eq, hit = START_EQ, None
        for d, v in enumerate(path, 1):
            eq = max(eq * (1 + L * v), 0.0)
            if d % 30 == 0:
                eq += MONTHLY
            if hit is None and eq >= TARGET:
                hit = d
            if d == 24 * 30:
                below += eq < START_EQ + 24 * MONTHLY
        hits24 += hit is not None and hit <= 720
        hits36 += hit is not None
        months.append(hit / 30 if hit else np.inf)
    return {"10k<=24mo": hits24 / n_paths, "10k<=36mo": hits36 / n_paths,
            "median months": float(np.median(months)), "P(below deposits at 24mo)": below / n_paths}


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    market = load_market_bulk("4h", last_month, UNIVERSE, lambda s: available_months(s, "4h"))
    end_ms = max(b["timestamp"] for m in market.values() for b in m["bars"]) + 1
    books = {
        "LONG 1% (live)": replace(LIVE),
        "SHORT 1%": replace(LIVE, allow_long=False, allow_short=True),
        "LONG 1% + SHORT 1%": replace(LIVE, allow_short=True),
        "LONG 1% + SHORT 0.5%": replace(LIVE, allow_short=True, short_risk_pct=0.005),
        "LONG 1% + SHORT 0.25%": replace(LIVE, allow_short=True, short_risk_pct=0.0025),
    }
    streams: Dict[str, pd.Series] = {}
    print("In-sample 2020-06 .. 2024-06 | out-of-sample 2024-07 .. now")
    for name, p in books.items():
        curve: list = []
        s = simulate(market, p, _ms(START), end_ms, curve_out=curve)
        eq = pd.Series(dict(curve))
        eq.index = pd.to_datetime(eq.index, unit="ms")
        streams[name] = eq.resample("1D").last().dropna().pct_change().dropna()
        print(fmt(name, streams[name]) + f" | trades {s['trades']}", flush=True)
    df = pd.DataFrame(streams).dropna()
    print("\nCorrelation of daily returns, long book vs short book: "
          f"{df['LONG 1% (live)'].corr(df['SHORT 1%']):+.2f}")
    print("Yearly returns:")
    yearly = (1 + df).groupby(df.index.year).prod() - 1
    print(yearly.map(lambda v: f"{v:+.0%}").to_string())

    print(f"\nDeposit plan: {START_EQ:.0f} to start + {MONTHLY:.0f}/month -> {TARGET:,.0f}")
    for y in (0.0, 0.15, 0.30, 0.50, 0.86):
        m = months_to_target(y)
        print(f"  at {y:+.0%}/yr: {m} months ({m / 12:.1f} years), {START_EQ + m * MONTHLY:,.0f} deposited")

    print("\nHistorical odds with deposits (fresh account every 2 weeks since 2020-06):")
    rows = []
    for name in ("LONG 1% (live)", "LONG 1% + SHORT 0.5%", "LONG 1% + SHORT 1%"):
        for L in (1, 2, 3, 5):
            rows.append({"book": name, "risk/trade": f"{L:.0f}%", **deposit_odds(df[name], L)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nStress test: 36-month paths stitched from random 30-day blocks of the out-of-sample period only "
          f"({SPLIT} .. now, +30%/yr at 1% risk):")
    rows = []
    oos = df["LONG 1% (live)"][SPLIT:]
    for L in (1, 2, 3, 5):
        rows.append({"risk/trade": f"{L:.0f}%", **bootstrap_odds(oos, L)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
