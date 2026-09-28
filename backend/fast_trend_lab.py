"""
A faster crypto trend book: the live breakout rules on 1-hour bars instead of 4-hour bars, traded with limit (maker)
orders so the extra trades cost less. The hope is a second crypto book that catches shorter moves and is only partly
correlated with the 4h book, the way the TradFi book raised the plan's Sharpe. Fixed in advance:

  rules    the LIVE rules (long-only, Donchian entry / exit channel, 4 x ATR trailing stop, 1% risk per trade) with
           the same bar counts on 1h bars: 120 / 60 bars = 5 / 2.5 days (primary); 240 / 120 = 10 / 5 days (check)
  orders   maker_lab.simulate "limit_chase": entries and channel exits as limits at the signal close for one bar
           (0.02% maker), then a market order; stops are market orders (0.05% + 0.03% slippage)
  test     the book alone, its correlation with the 4h book, and plan + fast book (equal risk, weights from the
           in-sample period) vs the plan; in-sample 2020-06 .. 2024-06, out-of-sample 2024-07 .. now
  keep     only if plan + fast book beats the plan's Sharpe in BOTH periods

    python -m backend.fast_trend_lab
"""

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from backend.backtest_trend import UNIVERSE, _ms
from backend.cross_asset_lab import BINANCE_TRADFI, FUNDING_2026, etf_prices, tsmom_returns
from backend.growth_study import LIVE, load_market_bulk
from backend.maker_lab import simulate
from backend.strategy_lab import START, fmt
from backend.universe_data import available_months

SPLIT = "2024-06-30"


def book(interval: str, p, mode: str = "limit_chase"):
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    market = load_market_bulk(interval, last_month, UNIVERSE, lambda s: available_months(s, interval))
    end = max(b["timestamp"] for m in market.values() for b in m["bars"]) + 1
    curve, st = simulate(market, p, _ms(START), end, mode)
    eq = pd.Series(dict(curve))
    eq.index = pd.to_datetime(eq.index, unit="ms")
    return eq.resample("1D").last().dropna().pct_change().dropna(), st


def main():
    runs = {
        "4h 120/60 (live), limit": ("4h", LIVE, "limit_chase"),
        "1h 120/60, limit": ("1h", replace(LIVE, interval="1h"), "limit_chase"),
        "1h 120/60, market orders": ("1h", replace(LIVE, interval="1h"), "taker"),
        "1h 240/120, limit": ("1h", replace(LIVE, interval="1h", entry_n=240, exit_n=120), "limit_chase"),
    }
    books = {}
    for name, (iv, p, mode) in runs.items():
        t0 = time.time()
        r, st = book(iv, p, mode)
        books[name] = r
        print(f"{name:28s} entries {st['entries']:5d} (maker {st['maker_entries']:5d}), fees ${st['fees']:,.0f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    days = pd.date_range(max(b.index.min() for b in books.values()), min(b.index.max() for b in books.values()), freq="D")
    b = pd.DataFrame(books).reindex(days).fillna(0.0)
    slow = b["4h 120/60 (live), limit"]
    print("\nIn-sample 2020-06 .. 2024-06 | out-of-sample 2024-07 .. now (each book on its own $500 account)")
    for name in b:
        print(fmt(name, b[name]))
    for name in list(b)[1:]:
        print(f"correlation of '{name}' with the 4h book: in {b[name][:SPLIT].corr(slow[:SPLIT]):+.2f} | "
              f"out {b[name]['2024-07-01':].corr(slow['2024-07-01':]):+.2f}")

    tb = tsmom_returns(etf_prices("2005-01-01")[list(BINANCE_TRADFI)], FUNDING_2026).reindex(days).fillna(0.0)
    print("\nPlan variants, scaled to the 4h book's in-sample volatility:")
    variants = [("plan (4h crypto + TradFi)", {"c": slow, "t": tb})]
    for name in list(b)[1:]:
        variants.append((f"plan + {name}", {"c": slow, "t": tb, "f": b[name]}))
    variants.append(("plan with 1h 120/60 replacing 4h", {"f": b["1h 120/60, limit"], "t": tb}))
    for name, cols in variants:
        both = pd.DataFrame(cols)
        vol_is = both[START:SPLIT].std()
        w = (1 / vol_is) / (1 / vol_is).sum()
        combo = (both * w).sum(axis=1)
        combo *= slow[START:SPLIT].std() / combo[START:SPLIT].std()
        print(fmt(name, combo))


if __name__ == "__main__":
    main()
