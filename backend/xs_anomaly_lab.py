"""
Two more cross-sectional ideas on the point-in-time top 30 Binance USDT perps, traded like positioning_lab's book
(7 daily slices, long 6 / short 6, 0.1% per unit of turnover, real funding, 2022-23 vs 2024+). Fixed in advance:

  lottery    coins that recently had a huge up-day are overpriced (the "MAX effect": Bali, Cakici & Whitelaw 2011
             in stocks; also reported in crypto). Signal = minus the largest daily return of the past 30 days
             (long the dullest coins, short the lottery tickets)
  crowding   lots of open futures positions relative to trading volume means crowded leverage that unwinds.
             Signal = minus log(open interest value / 30-day mean daily quote volume)
  pass       Sharpe > 0.5 with funding in both periods, and still positive with the past 30-day return removed

    python -m backend.xs_anomaly_lab
"""

import time

import numpy as np
import pandas as pd

from backend import positioning_lab as pl
from backend.universe_data import daily_panel


def main():
    sig, ret, member, close = pl.signals()
    idx = ret.index
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    qvol = daily_panel(last_month)["qvol"].reindex(index=idx, columns=ret.columns)
    m = pl.metrics()
    oi = m.pivot_table(index="day", columns="sym", values="oi_value").reindex(index=idx, columns=ret.columns)
    rclose = close.pct_change(fill_method=None)
    signals = {
        "lottery (minus max daily return, 30d)": -rclose.rolling(30, min_periods=20).max().reindex(idx),
        "crowding (minus OI / volume)": -np.log(oi / qvol.rolling(30, min_periods=20).mean()),
    }
    held = sorted(set(ret.columns[member.any().to_numpy()]))
    fund = pl.daily_funding(held, idx).reindex(columns=ret.columns).fillna(0.0)
    past = (close / close.shift(30) - 1).reindex(idx)
    z = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

    def book(s, **kw):
        return pd.concat([pl.backtest(s, ret, member, offset=o, **kw) for o in range(7)], axis=1).mean(axis=1)
    for name, s in signals.items():
        zs, zp = z(s.where(member)), z(past.where(member))
        beta = (zs * zp).sum(axis=1) / (zp * zp).sum(axis=1)
        resid = zs - zp.mul(beta, axis=0)
        print(f"\n{name}: correlation with the past 30-day return {zs.corrwith(zp, axis=1).mean():+.2f}, "
              f"with the smart-money signal {zs.corrwith(z(sig['smart'].where(member)), axis=1).mean():+.2f}")
        print(f"  {'price only':30s} {pl._fmt(book(s))}")
        print(f"  {'with funding':30s} {pl._fmt(book(s, funding=fund))}")
        print(f"  {'momentum removed, funding':30s} {pl._fmt(book(resid, funding=fund))}")
        print(f"  {'long leg vs average':30s} {pl._fmt(book(s, side='long'))}")


if __name__ == "__main__":
    main()
