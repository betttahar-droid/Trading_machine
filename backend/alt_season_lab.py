"""
"Alt season" filter for the live trend strategy: take breakout entries in the 7 altcoins only while an equal-weight
index of those altcoins is above its 30-day average when priced in BTC (altcoins outperforming Bitcoin); BTC entries
and all exits unchanged. One setting, fixed in advance; in-sample 2020-06 .. 2024-06, out-of-sample 2024-07 .. now.

    python -m backend.alt_season_lab
"""

import time

import numpy as np
import pandas as pd

from backend.backtest_trend import UNIVERSE, _ms
from backend.growth_study import LIVE, load_market_bulk
from backend.maker_lab import simulate
from backend.strategy_lab import START
from backend.universe_data import available_months

MA_BARS = 180                                   # 30 days of 4h bars


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    market = load_market_bulk("4h", last_month, UNIVERSE, lambda s: available_months(s, "4h"))
    end = max(b["timestamp"] for m in market.values() for b in m["bars"]) + 1
    close = pd.DataFrame({s: pd.Series({b["timestamp"]: b["close"] for b in m["bars"]}) for s, m in market.items()}).sort_index()
    alts = [s for s in close.columns if s != "BTCUSDT"]
    rel = (close[alts].pct_change(fill_method=None).mean(axis=1).fillna(0) - close["BTCUSDT"].pct_change().fillna(0))
    idx = (1 + rel).cumprod()
    season = (idx > idx.rolling(MA_BARS).mean()).to_dict()
    print(f"alt season on {np.mean(list(season.values())):.0%} of the time")

    def allow(sym, ts):
        return sym == "BTCUSDT" or bool(season.get(ts, True))
    for name, flt in (("current strategy", None), ("alt-season filter", allow)):
        curve, st = simulate(market, LIVE, _ms(START), end, "taker", allow=flt)
        eq = pd.Series(dict(curve))
        eq.index = pd.to_datetime(eq.index, unit="ms")
        r = eq.resample("1D").last().pct_change().dropna()
        out = []
        for lab, x in (("2020-06..2024-06", r[:"2024-06-30"]), ("2024-07..now", r["2024-07-01":])):
            e = (1 + x).cumprod()
            out.append(f"{lab} {e.iloc[-1] ** (365 / len(x)) - 1:+.1%}/yr Sharpe {x.mean() / x.std() * np.sqrt(365):+.2f} "
                       f"DD {(e / e.cummax() - 1).min():+.0%}")
        print(f"{name:18s} entries {st['entries']:4d} | " + " | ".join(out))


if __name__ == "__main__":
    main()
