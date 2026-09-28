"""
Gold / silver ratio mean reversion. The ratio (ounces of silver per ounce of gold) swings between ~40 and ~120 and
traders often bet on it returning to its middle. Both legs trade on Binance (XAUUSDT, XAGUSDT). Fixed in advance:

  ratio     log(GLD / SLV) (ETFs, 2006+), z-score vs its own past 3 years (756 trading days)
  rule      z > +1.5: long silver / short gold; z < -1.5: long gold / short silver; flat again once |z| < 0.25;
            equal dollar legs, each leg sized to 10% yearly volatility (60-day), 0.05% cost per unit of turnover
  funding   also shown with Binance's 2026 perp funding (longs pay gold 12.1%/yr, silver 23.0%/yr; shorts receive)
  periods   2009-2016 vs 2017-now; pass = Sharpe > 0.4 in both without funding and > 0 with funding after 2017
  check     an always-on version (position = -z / 2, capped at +-1)

    python -m backend.gold_silver_lab
"""

import numpy as np
import pandas as pd

from backend.cross_asset_lab import FUNDING_2026, etf_prices

COST = 0.0005


def run(px: pd.DataFrame, rule: str, funding: bool) -> pd.Series:
    lr = np.log(px["GLD"] / px["SLV"])
    z = (lr - lr.rolling(756, min_periods=500).mean()) / lr.rolling(756, min_periods=500).std()
    ret = px[["GLD", "SLV"]].pct_change(fill_method=None)
    vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    if rule == "threshold":
        side, pos = 0.0, []
        for v in z.to_numpy():
            if np.isnan(v):
                pos.append(0.0)
                continue
            if side == 0 and v > 1.5:
                side = -1.0                       # ratio high: short gold, long silver
            elif side == 0 and v < -1.5:
                side = 1.0                        # ratio low: long gold, short silver
            elif side != 0 and abs(v) < 0.25:
                side = 0.0
            pos.append(side)
        s = pd.Series(pos, index=z.index)
    else:
        s = (-z / 2).clip(-1, 1).fillna(0.0)
    w = pd.DataFrame({"GLD": s * 0.10 / vol["GLD"], "SLV": -s * 0.10 / vol["SLV"]}).shift(1).fillna(0.0)
    r = (w * ret.fillna(0)).sum(axis=1) - (w.diff().abs().sum(axis=1) * COST)
    if funding:
        r = r - (w["GLD"] * FUNDING_2026["GLD"] + w["SLV"] * FUNDING_2026["SLV"]) / 252
    return r


def main():
    px = etf_prices("2004-01-01")[["GLD", "SLV"]].dropna()
    lr = np.log(px["GLD"] / px["SLV"])
    print(f"GLD/SLV log ratio {lr.index[0]:%Y-%m} .. {lr.index[-1]:%Y-%m}")
    for rule in ("threshold", "always-on"):
        for fund in (False, True):
            r = run(px, rule, fund)
            cells = []
            for lab, x in (("2009-16", r["2009":"2016"]), ("2017+", r["2017":])):
                eq = (1 + x).cumprod()
                cells.append(f"{lab} {eq.iloc[-1] ** (252 / len(x)) - 1:+6.1%}/yr Sharpe {x.mean() / x.std() * np.sqrt(252):+.2f} "
                             f"DD {(eq / eq.cummax() - 1).min():+.0%}")
            active = (r["2009":] != 0).mean()
            print(f"  {rule:10s} {'Binance funding' if fund else 'no funding':15s} " + " | ".join(cells) + f" | active {active:.0%}")


if __name__ == "__main__":
    main()
