"""
Small-coin short-term reversal: do yesterday's biggest losers bounce and biggest pumps fade?

Point-in-time universe: the top N USDT perps by trailing 30-day volume (delisted coins included, see
universe_data.py). At each daily close, rank coins by their return over the last `lookback` days; buy the
bottom decile (losers) and/or short the top decile (winners), equal weight, hold `hold` days (overlapping
books, 1/hold of the capital each). Costs: taker fee 0.05% + slippage 0.10% per side on every change in weight.
Funding is ignored (it mostly favours this trade: pumped coins pay shorts, dumped coins pay longs).

Settings are compared on 2020-06 .. 2024-06; 2024-07 .. now is the out-of-sample check.

    python -m backend.smallcap_lab
"""

import time

import numpy as np
import pandas as pd

from backend.backtest_trend import SPLIT
from backend.strategy_lab import START, fmt
from backend.universe_data import daily_panel, top_by_volume

COST = 0.0005 + 0.001


def reversal_returns(close: pd.DataFrame, member: pd.DataFrame, lookback: int, hold: int, side: str,
                     frac: float = 0.1) -> pd.Series:
    ret = close.pct_change(fill_method=None)
    past = close / close.shift(lookback) - 1
    rank = past.where(member).rank(axis=1, pct=True)
    losers = (rank <= frac).astype(float)
    winners = (rank > 1 - frac).astype(float)
    book = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if side in ("long", "both"):
        book += losers.div(losers.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if side in ("short", "both"):
        book -= winners.div(winners.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if side == "both":
        book /= 2
    w = sum(book.shift(k) for k in range(1, hold + 1)) / hold      # overlapping books, decided at the close
    gross = (w * ret.fillna(0.0)).sum(axis=1)
    cost = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1) * COST
    return (gross - cost)[START:]


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    panel = daily_panel(last_month)
    close = panel["close"]
    for n in (50, 100):
        member = top_by_volume(panel["qvol"], close, n)
        print(f"\nTop {n} by volume. In-sample 2020-06 .. 2024-06 | out-of-sample 2024-07 .. now", flush=True)
        for side in ("long", "short", "both"):
            for lookback in (1, 3):
                for hold in (1, 3):
                    r = reversal_returns(close, member, lookback, hold, side)
                    print("  " + fmt(f"{side:<5} lb={lookback} hold={hold}", r), flush=True)


if __name__ == "__main__":
    main()
