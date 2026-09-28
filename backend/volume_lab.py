"""
Abnormal trading volume across coins. In stocks, unusually high volume is followed by higher returns (the "high-volume
return premium", Gervais, Kaniel & Mingelgrin 2001: attention brings new buyers). Fixed in advance, on the
point-in-time top 30 Binance USDT perps like positioning_lab's book:

  signal    log(mean quote volume of the last 7 days / mean of the 60 days before): long the coins with the biggest
            volume surge, short the quietest (the literature's sign; the opposite is shown as a check)
  book      7 daily slices, long 6 / short 6, 0.1% per unit of turnover, real funding
  pass      Sharpe > 0.5 with funding in both 2022-23 and 2024+, and still positive with the past 7-day return removed

Also: how long the smart-money signal lasts (holding 1 .. 28 days, as many slices as days).

    python -m backend.volume_lab
"""

import time

import numpy as np
import pandas as pd

from backend import positioning_lab as pl
from backend.universe_data import daily_panel


def main():
    sig, ret, member, close = pl.signals()
    idx = ret.index
    held = sorted(set(ret.columns[member.any().to_numpy()]))
    fund = pl.daily_funding(held, idx).reindex(columns=ret.columns).fillna(0.0)

    def book(s, every=7, **kw):
        return pd.concat([pl.backtest(s, ret, member, offset=o, every=every, **kw) for o in range(every)],
                         axis=1).mean(axis=1)
    print("How long the smart-money signal lasts (with funding):")
    for hold in (1, 3, 7, 14, 28):
        print(f"  hold {hold:2d} days {pl._fmt(book(sig['smart'], every=hold, funding=fund))}")

    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    qvol = daily_panel(last_month)["qvol"].reindex(index=idx, columns=ret.columns)
    surge = np.log(qvol.rolling(7, min_periods=5).mean() / qvol.shift(7).rolling(60, min_periods=40).mean())
    past = (close / close.shift(7) - 1).reindex(idx)
    z = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
    zs, zp = z(surge.where(member)), z(past.where(member))
    beta = (zs * zp).sum(axis=1) / (zp * zp).sum(axis=1)
    resid = zs - zp.mul(beta, axis=0)
    print(f"\nAbnormal volume: correlation with the past 7-day return {zs.corrwith(zp, axis=1).mean():+.2f}, "
          f"with the smart-money signal {zs.corrwith(z(sig['smart'].where(member)), axis=1).mean():+.2f}")
    print(f"  {'volume surge, price only':32s} {pl._fmt(book(surge))}")
    print(f"  {'volume surge, with funding':32s} {pl._fmt(book(surge, funding=fund))}")
    print(f"  {'momentum removed, with funding':32s} {pl._fmt(book(resid, funding=fund))}")
    print(f"  {'opposite sign (check)':32s} {pl._fmt(book(-surge, funding=fund))}")


if __name__ == "__main__":
    main()
