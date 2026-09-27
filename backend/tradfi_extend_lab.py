"""
More markets for the TradFi trend book: Binance added perps on crude oil (CLUSDT, BZUSDT), copper, platinum,
palladium and the Russell 2000 (IWMUSDT) in 2026. Does trading them with the same 12-month trend rule raise the
plan's Sharpe, the way adding gold / silver / S&P / Nasdaq did?

History from ETFs: USO (WTI), CPER (copper), PPLT (platinum), PALL (palladium), IWM (Russell 2000), next to the current
GLD, SLV, SPY, QQQ. Perp funding (longs pay, per year of notional, Binance 2026):
  strict      every asset pays its 2026 funding if positive; negative funding (oil) is ignored (never a gain)
  realistic   physical-metal and equity ETFs pay the perp funding; futures-based ETFs (USO, CPER) already include
              their roll carry, so no extra funding
Same rule, weights and scaling as cross_asset_lab.py; plan = equal-risk crypto trend + TradFi book.

    python -m backend.tradfi_extend_lab
"""

import numpy as np
import pandas as pd

from backend.cross_asset_lab import FUNDING_2026, crypto_trend_returns, mixed_bootstrap, tsmom_returns
from backend.strategy_lab import START, fmt

CURRENT = ["GLD", "SLV", "SPY", "QQQ"]
NEW = ["IWM", "USO", "CPER", "PPLT", "PALL"]
FUND_NEW = {"IWM": -0.002, "USO": -0.22, "CPER": 0.134, "PPLT": 0.181, "PALL": 0.205}
FUTURES_ETF = {"USO", "CPER"}


def main():
    import yfinance as yf
    px = yf.download(CURRENT + NEW, start="2005-01-01", progress=False, auto_adjust=True)["Close"].dropna(how="all")
    strict = {k: max(v, 0.0) for k, v in {**FUNDING_2026, **FUND_NEW}.items()}
    realistic = {**FUNDING_2026, **{k: (0.0 if k in FUTURES_ETF else v) for k, v in FUND_NEW.items()}}
    books = {"current 4 (strict funding)": tsmom_returns(px[CURRENT], strict),
             "9 markets, strict funding": tsmom_returns(px[CURRENT + NEW], strict),
             "9 markets, realistic funding": tsmom_returns(px[CURRENT + NEW], realistic),
             "9 minus platinum/palladium": tsmom_returns(px[CURRENT + ["IWM", "USO", "CPER"]], strict)}
    print("TradFi trend book alone, 2012 .. now (all 9 ETFs trade from 2011-11):")
    for name, b in books.items():
        x = b["2012-01-01":]
        eq = (1 + x).cumprod()
        print(f"  {name:32s} {eq.iloc[-1] ** (252 / len(x)) - 1:+.1%}/yr  Sharpe {x.mean() / x.std() * np.sqrt(252):+.2f}  "
              f"max DD {(eq / eq.cummax() - 1).min():+.0%}")

    crypto = crypto_trend_returns()
    days = pd.date_range(crypto.index.min(), crypto.index.max(), freq="D")
    cd = crypto.reindex(days).fillna(0.0)
    print("\nPlan = crypto trend + TradFi book, equal risk (weights from 2020-06 .. 2024-06), scaled to crypto's volatility")
    print("In-sample 2020-06 .. 2024-06 | out-of-sample 2024-07 .. now")
    print(fmt("crypto trend alone", cd))
    for name, b in books.items():
        both = pd.DataFrame({"c": cd, "e": b.reindex(days).fillna(0.0)})
        vol_is = both[START:"2024-06-30"].std()
        w = (1 / vol_is) / (1 / vol_is).sum()
        combo = (both * w).sum(axis=1)
        combo *= cd[START:"2024-06-30"].std() / combo[START:"2024-06-30"].std()
        print(fmt(f"plan with {name}", combo) + f"  (corr {both.corr().iloc[0, 1]:+.2f})")
        long_hist = b["2012-01-01":]
        cal = long_hist.reindex(pd.date_range(long_hist.index.min(), long_hist.index.max(), freq="D")).fillna(0.0)
        scale = cd[START:"2024-06-30"].std() / (both * w).sum(axis=1)[START:"2024-06-30"].std()
        odds = mixed_bootstrap(cd["2024-07-01":], cal, w, scale, 2)
        print(f"      level 2, 500 + 100/month: 10k within 36 months {odds['10k<=36mo']:.0%}, "
              f"below deposits at 24 months {odds['P(below deposits at 24mo)']:.0%}")


def tsmom_long_short(px: pd.DataFrame, funding: dict, short: bool = True) -> pd.Series:
    """cross_asset_lab.tsmom_returns with shorts when the 12-month return is negative; funding is paid by longs and
    received by shorts (and the reverse when negative)."""
    from backend.cross_asset_lab import COST, TARGET_VOL
    ret = px.pct_change(fill_method=None)
    mom = px / px.shift(252) - 1
    vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    month_end = px.index.to_series().groupby(px.index.to_period("M")).transform("max") == px.index.to_series()
    n = px.notna().sum(axis=1).clip(lower=1)
    side = np.sign(mom) if short else (mom > 0).astype(float)
    target = (side * (TARGET_VOL / np.sqrt(n)).values[:, None] / vol).where(month_end)
    w = target.ffill().fillna(0.0).clip(-1.0, 1.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * ret.fillna(0.0)).sum(axis=1)
    cost = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1) * COST
    cost = cost + (held * pd.Series(funding).reindex(held.columns).fillna(0.0) / 252).sum(axis=1)
    return gross - cost


def long_short():
    import yfinance as yf
    from backend.cross_asset_lab import crypto_trend_returns
    px = yf.download(CURRENT + NEW, start="2005-01-01", progress=False, auto_adjust=True)["Close"].dropna(how="all")
    realistic = {**FUNDING_2026, **{k: (0.0 if k in FUTURES_ETF else v) for k, v in FUND_NEW.items()}}
    crypto = crypto_trend_returns()
    days = pd.date_range(crypto.index.min(), crypto.index.max(), freq="D")
    cd = crypto.reindex(days).fillna(0.0)
    print("\nLong / short (Moskowitz-Ooi-Pedersen) vs long / flat, realistic funding")
    for label, cols in (("current 4", CURRENT), ("9 markets", CURRENT + NEW), ("current 4 + oil + copper", CURRENT + ["USO", "CPER"])):
        for short in (False, True):
            b = tsmom_long_short(px[cols], realistic, short)
            x = b["2012-01-01":]
            eq = (1 + x).cumprod()
            both = pd.DataFrame({"c": cd, "e": b.reindex(days).fillna(0.0)})
            vol_is = both[START:"2024-06-30"].std()
            w = (1 / vol_is) / (1 / vol_is).sum()
            combo = (both * w).sum(axis=1)
            combo *= cd[START:"2024-06-30"].std() / combo[START:"2024-06-30"].std()
            print(f"  {label:26s} {'long/short' if short else 'long/flat ':10s} book 2012+: Sharpe {x.mean() / x.std() * np.sqrt(252):+.2f}, "
                  f"max DD {(eq / eq.cummax() - 1).min():+.0%} || plan: {fmt('', combo).strip()}")


if __name__ == "__main__":
    import sys
    long_short() if sys.argv[1:] == ["longshort"] else main()
