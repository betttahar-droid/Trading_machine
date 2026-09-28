"""
Does the smart-money signal (positioning_lab.py) improve the live trend strategy's entries? Fixed in advance:

  filter   skip a breakout entry when the coin's smart-money gap (3-day mean of log top-trader position ratio - log
           all-account ratio, known at the previous UTC close) is below the median of that day's top-30 perps
  compare  unfiltered, and skipping the same share of entries at random (5 seeds); 2022-01 .. now (the metrics start
           in 2021-12), 2022-23 vs 2024+; the live 8 coins and rules, market orders (maker_lab.simulate "taker")
  pass     filtered Sharpe above unfiltered and above every random-skip run in both periods

    python -m backend.smart_filter_lab
"""

import time

import numpy as np
import pandas as pd

from backend import positioning_lab as pl
from backend.backtest_trend import UNIVERSE, _ms
from backend.growth_study import LIVE, load_market_bulk
from backend.maker_lab import simulate
from backend.universe_data import available_months

DAY = 86_400_000


def run(market, allow):
    end = max(b["timestamp"] for m in market.values() for b in m["bars"]) + 1
    curve, st = simulate(market, LIVE, _ms("2022-01-01"), end, "taker", allow=allow)
    eq = pd.Series(dict(curve))
    eq.index = pd.to_datetime(eq.index, unit="ms")
    return eq.resample("1D").last().dropna().pct_change().dropna(), st


def sharpe(r):
    return " | ".join(f"{lab} Sharpe {x.mean() / x.std() * np.sqrt(365):+.2f} CAGR {(1 + x).prod() ** (365 / len(x)) - 1:+.0%}"
                      for lab, x in (("2022-23", r[:"2023-12-31"]), ("2024+", r["2024-01-01":])))


def main():
    sig, ret, member, _ = pl.signals()
    smart = sig["smart"]
    ok = smart.ge(smart.where(member).median(axis=1), axis=0) | smart.isna()   # no data -> allowed
    ok_by_day = {d.value // 10**6: row for d, row in ok.shift(1).fillna(True).iterrows()}
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    market = load_market_bulk("4h", last_month, UNIVERSE, lambda s: available_months(s, "4h"))
    seen = {"asked": 0, "skipped": 0}

    def allow(sym, ts):
        row = ok_by_day.get(ts // DAY * DAY)
        good = True if row is None or sym not in row.index else bool(row[sym])
        seen["asked"] += 1
        seen["skipped"] += not good
        return good
    base, st0 = run(market, None)
    filt, st1 = run(market, allow)
    share = seen["skipped"] / max(seen["asked"], 1)
    print(f"entries unfiltered {st0['entries']}, filtered {st1['entries']} (signals skipped {share:.0%})")
    print(f"  unfiltered            {sharpe(base)}")
    print(f"  smart-money filter    {sharpe(filt)}")
    for seed in range(5):
        rng = np.random.default_rng(seed)
        r, _ = run(market, lambda s, t: rng.random() >= share)
        print(f"  random skip seed {seed}    {sharpe(r)}")


if __name__ == "__main__":
    main()
