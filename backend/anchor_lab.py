"""
Two last checks on data already collected. Fixed in advance:

1  Market-wide smart money as a brake for the crypto trend book: when Binance's top traders lean short against the
   crowd across the whole top 30 at once, cut risk. Aggregate = median over the top-30 coins of the 3-day smart-money
   gap; z-score vs its own past 180 days; the trend book runs at half size the next day while z < -1.
   Pass = better Sharpe than the plain trend book in both 2022-23 and 2024+.

2  52-week-high anchoring (George & Hwang 2004, stocks): coins near their one-year high keep outperforming, those far
   below keep lagging. Signal = close / highest close of the past 365 days; top-30 7-slice long 6 / short 6 with
   funding. Pass = Sharpe > 0.5 in both periods and still positive with the past 30-day return removed.

    python -m backend.anchor_lab
"""

import numpy as np
import pandas as pd

from backend import positioning_lab as pl
from backend.cross_asset_lab import crypto_trend_returns


def main():
    sig, ret, member, close = pl.signals()
    agg = sig["smart"].where(member).median(axis=1)
    z = (agg - agg.rolling(180, min_periods=90).mean()) / agg.rolling(180, min_periods=90).std()
    trend = crypto_trend_returns()
    scale = pd.Series(np.where(z.shift(1) < -1, 0.5, 1.0), index=z.index).reindex(trend.index).fillna(1.0)
    braked = trend * scale
    print("1) Trend book with the market-wide smart-money brake (half size while z < -1):")
    for lab, r in (("plain trend book", trend), ("with the brake", braked)):
        cells = [f"{p} Sharpe {x.mean() / x.std() * np.sqrt(365):+.2f}"
                 for p, x in (("2022-23", r["2022":"2023"]), ("2024+", r["2024":]))]
        print(f"   {lab:18s} " + " | ".join(cells))
    print(f"   share of days braked: {(scale['2022':] < 1).mean():.0%}")

    held = sorted(set(ret.columns[member.any().to_numpy()]))
    fund = pl.daily_funding(held, ret.index).reindex(columns=ret.columns).fillna(0.0)
    anchor = (close / close.rolling(365, min_periods=200).max()).reindex(ret.index)
    past30 = (close / close.shift(30) - 1).reindex(ret.index)
    zf = lambda x: x.where(member).sub(x.where(member).mean(axis=1), axis=0).div(x.where(member).std(axis=1), axis=0)
    za, zp = zf(anchor), zf(past30)
    beta = (za * zp).sum(axis=1) / (zp * zp).sum(axis=1)
    book = lambda s, **kw: pd.concat([pl.backtest(s, ret, member, offset=o, **kw) for o in range(7)], axis=1).mean(axis=1)
    print("\n2) 52-week-high anchoring (top 30, 7 slices):")
    print(f"   correlation with the past 30-day return {za.corrwith(zp, axis=1).mean():+.2f}")
    print(f"   {'price only':28s} {pl._fmt(book(anchor))}")
    print(f"   {'with funding':28s} {pl._fmt(book(anchor, funding=fund))}")
    print(f"   {'momentum removed, funding':28s} {pl._fmt(book(za - zp.mul(beta, axis=0), funding=fund))}")


if __name__ == "__main__":
    main()
