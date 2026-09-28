"""
A better trend signal for the TradFi book? The plan holds gold / silver / S&P 500 / Nasdaq while their 12-month
return is positive. The long-history trend literature (Hurst, Ooi & Pedersen 2017, "A Century of Evidence on
Trend-Following Investing") averages 1-, 3- and 12-month signals, which reacts faster in reversals like 2008 and 2022.
Fixed in advance (published variants, nothing fitted here):

  12m        current rule: long while the 12-month return is positive (monthly)
  1/3/12     average of the three signs (0, 1/3, 2/3 or 1 of full size), monthly
  1/3/12 wk  the same, rebalanced weekly (Fridays)
  3/6/12     average of 3-, 6- and 12-month signs, monthly
Same volatility sizing, 0.05% costs and 2026 perp funding as cross_asset_lab.tsmom_returns.

  judged on  the book alone 2007 .. 2019 (before the plan's test years) and 2020 .. now, and plan (crypto trend +
             TradFi, equal risk) in-sample 2020-06 .. 2024-06 / out-of-sample 2024-07 .. now; adopt only if the
             book is better in 2007-19 AND the plan is better in both periods

    python -m backend.tradfi_signal_lab
"""

import numpy as np
import pandas as pd

from backend.cross_asset_lab import BINANCE_TRADFI, COST, FUNDING_2026, TARGET_VOL, crypto_trend_returns, etf_prices
from backend.strategy_lab import START, fmt


def trend_book(px: pd.DataFrame, lookbacks=(252,), weekly: bool = False, funding: dict = FUNDING_2026) -> pd.Series:
    ret = px.pct_change(fill_method=None)
    sig = sum((px / px.shift(lb) - 1 > 0).astype(float) for lb in lookbacks) / len(lookbacks)
    sig = sig.where(px.shift(max(lookbacks)).notna())
    vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    idx = px.index.to_series()
    if weekly:
        when = idx.groupby(idx.dt.to_period("W-FRI")).transform("max") == idx
    else:
        when = idx.groupby(idx.dt.to_period("M")).transform("max") == idx
    n = px.notna().sum(axis=1).clip(lower=1)
    target = (sig * (TARGET_VOL / np.sqrt(n)).values[:, None] / vol).where(when)
    w = target.ffill().fillna(0.0).clip(upper=1.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * ret.fillna(0.0)).sum(axis=1)
    cost = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1) * COST
    cost = cost + (held * pd.Series(funding).reindex(held.columns).fillna(0.0) / 252).sum(axis=1)
    return gross - cost


def main():
    px = etf_prices("2004-01-01")[list(BINANCE_TRADFI)]
    variants = {"12m (current)": dict(lookbacks=(252,)),
                "1/3/12 monthly": dict(lookbacks=(21, 63, 252)),
                "1/3/12 weekly": dict(lookbacks=(21, 63, 252), weekly=True),
                "3/6/12 monthly": dict(lookbacks=(63, 126, 252))}
    books = {k: trend_book(px, **v) for k, v in variants.items()}
    print("TradFi book alone (trading days):")
    for k, b in books.items():
        out = []
        for lab, x in (("2007-19", b["2007":"2019"]), ("2020-now", b["2020":])):
            eq = (1 + x).cumprod()
            out.append(f"{lab} {eq.iloc[-1] ** (252 / len(x)) - 1:+6.1%}/yr Sharpe {x.mean() / x.std() * np.sqrt(252):+.2f} "
                       f"DD {(eq / eq.cummax() - 1).min():+.0%}")
        yr = (1 + b["2007":]).groupby(b["2007":].index.year).prod() - 1
        print(f"  {k:16s} " + " | ".join(out))
        print(f"  {'':16s} 2008 {yr[2008]:+.0%}, 2018 {yr[2018]:+.0%}, 2020 {yr[2020]:+.0%}, 2022 {yr[2022]:+.0%}")

    crypto = crypto_trend_returns()
    days = pd.date_range(crypto.index.min(), crypto.index.max(), freq="D")
    cd = crypto.reindex(days).fillna(0.0)
    print("\nPlan (crypto trend + TradFi, equal risk), in-sample 2020-06 .. 2024-06 | out-of-sample 2024-07 .. now:")
    for k, b in books.items():
        tb = b.reindex(days).fillna(0.0)
        both = pd.DataFrame({"c": cd, "t": tb})
        vol_is = both[START:"2024-06-30"].std()
        w = (1 / vol_is) / (1 / vol_is).sum()
        combo = (both * w).sum(axis=1)
        combo *= cd[START:"2024-06-30"].std() / combo[START:"2024-06-30"].std()
        print(fmt(f"plan, TradFi {k}", combo))


if __name__ == "__main__":
    main()
