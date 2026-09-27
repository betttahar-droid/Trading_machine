"""
A closer look at shorting new Binance USDT perpetual listings (backend/listing_lab.py), the one idea that kept
working after 2024-07 (+2.3% to +3.3% per trade on ~550 listings, but nothing before).

  1 market control  each trade minus shorting an equal-weight basket of established perps (listed > 1 year, in the
                    top 50 by volume) over the same days: is it new listings, or just the altcoin bear market?
  2 stability       per quarter since 2024-07
  3 portfolio       every new listing shorted with 2% of equity (daily marked, stop, funding, fees), all at once:
                    yearly return, Sharpe, worst drawdown, worst month

Setting: wait 3 days after listing, hold 21 days, stop at +60% (the middle of listing_lab's grid, not the best).

    python -m backend.listing_deep_lab
"""

import time

import numpy as np
import pandas as pd

from backend.listing_lab import FEE, SLIP, listing_trades
from backend.universe_data import available_months, daily_panel, load_funding, top_by_volume

WAIT, HOLD, STOP, SIZE = 3, 21, 0.60, 0.02


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    panel = daily_panel(last_month)
    close = panel["close"]
    cache = {}

    def funding_of(sym):
        if sym not in cache:
            f = load_funding(sym, available_months(sym, "1d")[:4])
            cache[sym] = f.resample("1D").sum() if len(f) else pd.Series(dtype=float)
        return cache[sym]

    t = listing_trades(panel, funding_of, WAIT, HOLD, STOP)
    t = t.sort_values("entry_day").reset_index(drop=True)

    # 1 market control: established perps (listed > 365 days, top 50 by trailing volume) over the same window
    member = top_by_volume(panel["qvol"], close, 50)
    age = close.notna().cumsum()
    ret = close.pct_change(fill_method=None)
    est = member & (age > 365)
    basket = ret.where(est).mean(axis=1)                         # equal-weight daily return of established coins
    ctrl = []
    for r in t.itertuples():
        i = close.index.get_loc(r.entry_day)
        seg = basket.iloc[i + 1:i + 1 + max(int(r.days), 1)]
        ctrl.append(-((1 + seg.fillna(0)).prod() - 1) - 2 * FEE - 2 * SLIP)
    t["ctrl"] = ctrl
    t["excess"] = t["ret"] - t["ctrl"]
    t["q"] = t["entry_day"].dt.to_period("Q")

    def s(x):
        x = x.dropna()
        return f"{100 * x.mean():+.2f}% (t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}, n {len(x)}, {100 * (x > 0).mean():.0f}% win)"
    print(f"{len(t)} listings shorted (wait {WAIT}d, hold {HOLD}d, stop +{STOP:.0%})")
    for lab, m in (("before 2024-07", t.entry_day < "2024-07-01"), ("2024-07 ..    ", t.entry_day >= "2024-07-01")):
        g = t[m]
        print(f"  {lab} short new listing {s(g.ret)} | short established basket {s(g.ctrl)} | difference {s(g.excess)}")
        print(f"      stopped out {100 * (g.reason == 'stop').mean():.0f}%, worst trade {100 * g.ret.min():+.0f}%")
    print("\n  per quarter (short new listing | difference vs established basket):")
    for q, g in t[t.entry_day >= "2023-01-01"].groupby("q"):
        print(f"    {q}: n {len(g):3d}  {100 * g.ret.mean():+6.2f}%  | {100 * g.excess.mean():+6.2f}%")

    # 3 portfolio: 2% of equity per short, marked to market daily
    days = close.index[close.index >= "2021-01-01"]
    pnl = pd.Series(0.0, index=days)
    for r in t.itertuples():
        sym = r.sym
        i = close.index.get_loc(r.entry_day)
        c = close[sym]
        entry = c.iloc[i] * (1 - SLIP)
        path = c.iloc[i:i + 1 + int(r.days)]
        daily = -(path.pct_change().fillna(0.0))                  # short's daily return on notional
        if r.reason == "stop":                                     # last day: realise the stop loss exactly
            prev = (entry - path.iloc[-2] * (1 + 0)) / entry if len(path) > 1 else 0.0
            daily.iloc[-1] = r.ret + 2 * FEE - daily.iloc[:-1].sum()
        daily.iloc[0] -= 2 * FEE + SLIP
        pnl = pnl.add(SIZE * daily.reindex(days).fillna(0.0), fill_value=0.0)
    for lab, part in (("2021-01 .. 2024-06", pnl[:"2024-06-30"]), ("2024-07 .. now    ", pnl["2024-07-01":])):
        eq = (1 + part).cumprod()
        monthly = (1 + part).groupby(part.index.to_period("M")).prod() - 1
        print(f"\n  portfolio {lab}: {eq.iloc[-1] ** (365 / len(part)) - 1:+.1%}/yr, Sharpe {part.mean() / part.std() * np.sqrt(365):+.2f}, "
              f"max DD {(eq / eq.cummax() - 1).min():+.0%}, worst month {monthly.min():+.1%}, best month {monthly.max():+.1%}")


if __name__ == "__main__":
    main()
