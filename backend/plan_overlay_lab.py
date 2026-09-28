"""
Two overlays on the plan (crypto trend + TradFi trend with 2026 funding), fixed in advance:

  adaptive weights  each month-end, weight the two books by 1 / their trailing 90-day volatility (the plan uses fixed
                    weights from 2020-06 .. 2024-06 volatility)
  drawdown brake    halve total exposure once the plan is 15% below its peak; back to full size at a new peak

In-sample 2020-06 .. 2024-06 | out-of-sample 2024-07 .. now; same scaling as cross_asset_lab.

    python -m backend.plan_overlay_lab
"""

import numpy as np
import pandas as pd

from backend.cross_asset_lab import BINANCE_TRADFI, FUNDING_2026, crypto_trend_returns, etf_prices, tsmom_returns
from backend.strategy_lab import START, fmt


def brake(r: pd.Series, dd_limit=0.15, cut=0.5) -> pd.Series:
    eq, peak, scale, out = 1.0, 1.0, 1.0, []
    for x in r.to_numpy():
        out.append(scale * x)
        eq *= 1 + scale * x
        if eq >= peak:
            peak, scale = eq, 1.0
        elif eq / peak - 1 <= -dd_limit:
            scale = cut
    return pd.Series(out, index=r.index)


def main():
    crypto = crypto_trend_returns()
    days = pd.date_range(crypto.index.min(), crypto.index.max(), freq="D")
    cd = crypto.reindex(days).fillna(0.0)
    tb = tsmom_returns(etf_prices("2005-01-01")[list(BINANCE_TRADFI)], FUNDING_2026).reindex(days).fillna(0.0)
    both = pd.DataFrame({"c": cd, "t": tb})
    vol_is = both[START:"2024-06-30"].std()
    w = (1 / vol_is) / (1 / vol_is).sum()
    fixed = (both * w).sum(axis=1)
    k = cd[START:"2024-06-30"].std() / fixed[START:"2024-06-30"].std()
    fixed *= k
    vol90 = both.rolling(90, min_periods=60).std()
    wd = (1 / vol90).div((1 / vol90).sum(axis=1), axis=0)
    wd = wd.where(pd.Series(days, index=days).dt.is_month_end, np.nan).ffill().shift(1).fillna(w)
    adaptive = (both * wd).sum(axis=1)
    adaptive *= cd[START:"2024-06-30"].std() / adaptive[START:"2024-06-30"].std()
    print("In-sample 2020-06 .. 2024-06 | out-of-sample 2024-07 .. now")
    print(fmt("plan, fixed weights (current)", fixed))
    print(fmt("plan, adaptive 90-day weights", adaptive))
    print(fmt("plan + drawdown brake (15%, halve)", brake(fixed)))
    print(fmt("plan + brake 10%", brake(fixed, 0.10)))
    print(fmt("plan + brake 20%", brake(fixed, 0.20)))
    print(f"adaptive crypto weight: mean {wd['c'].mean():.0%}, range {wd['c'].min():.0%} .. {wd['c'].max():.0%} (fixed {w['c']:.0%})")


if __name__ == "__main__":
    main()
