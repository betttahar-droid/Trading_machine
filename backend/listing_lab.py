"""
Short new listings: do newly listed Binance USDT perpetuals fall after the launch hype?

For every USDT perp listed since 2021 (delisted ones included, see universe_data.py): short at the close
of day `wait` after listing, hold `hold` days, exit early if the daily high reaches the stop
(entry x (1 + stop)). Taker fees + slippage on both sides; funding is received by the short when
positive and paid when negative. Returns are per trade on the short's notional.

Settings are compared on listings before 2024-07; later listings are the out-of-sample check.

    python -m backend.listing_lab
"""

import time

import numpy as np
import pandas as pd

from backend.backtest_trend import SPLIT
from backend.universe_data import available_months, daily_panel, load_funding

FEE, SLIP = 0.0005, 0.001      # new listings are thin: double the usual slippage


def listing_trades(panel, funding_of, wait: int, hold: int, stop: float) -> pd.DataFrame:
    close, high = panel["close"], panel["high"]
    rows = []
    for sym in close.columns:
        c = close[sym].dropna()
        if len(c) < wait + 2 or c.index[0] < pd.Timestamp("2021-01-01"):
            continue
        entry_day = c.index[wait]
        entry = c.iloc[wait] * (1 - SLIP)                     # short fill
        h = high[sym].reindex(c.index)
        f = funding_of(sym).reindex(c.index).fillna(0.0)
        exit_px, reason, k = None, "time", wait
        for k in range(wait + 1, min(len(c), wait + 1 + hold)):
            if h.iloc[k] >= entry * (1 + stop):
                exit_px, reason = max(entry * (1 + stop), c.iloc[k - 1]) * (1 + SLIP), "stop"
                break
        if exit_px is None:
            exit_px = c.iloc[k] * (1 + SLIP)
            reason = "time" if k == wait + hold else "delisted/end"
        fund = f.iloc[wait + 1:k + 1].sum()                    # short receives positive funding
        ret = (entry - exit_px) / entry - 2 * FEE + fund
        rows.append({"sym": sym, "listed": c.index[0], "entry_day": entry_day, "ret": ret, "reason": reason,
                     "days": k - wait})
    return pd.DataFrame(rows)


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    panel = daily_panel(last_month)
    cache = {}

    def funding_of(sym):
        if sym not in cache:
            f = load_funding(sym, available_months(sym, "1d")[:4])      # first months are all we need
            cache[sym] = f.resample("1D").sum() if len(f) else pd.Series(dtype=float)
        return cache[sym]

    split = pd.Timestamp(SPLIT)
    print("Short every new USDT perp: wait days after listing / hold days / stop-loss")
    print(f"{'wait':>4} {'hold':>4} {'stop':>5} | {'listings before 2024-07: n, mean, median, win%, stopped':<52} | "
          f"after 2024-07")
    for wait in (1, 3, 7):
        for hold in (14, 30):
            for stop in (0.3, 0.6):
                df = listing_trades(panel, funding_of, wait, hold, stop)
                parts = []
                for g in (df[df["listed"] < split], df[df["listed"] >= split]):
                    parts.append(f"n={len(g):3d} mean {g['ret'].mean():+6.1%} median {g['ret'].median():+6.1%} "
                                 f"win {np.mean(g['ret'] > 0):3.0%} stop {np.mean(g['reason'] == 'stop'):3.0%}")
                print(f"{wait:4d} {hold:4d} {stop:5.0%} | {parts[0]:<52} | {parts[1]}", flush=True)


if __name__ == "__main__":
    main()
