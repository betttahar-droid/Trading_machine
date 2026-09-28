"""
Can a simple model combining all the crowd / flow signals beat the smart-money signal alone? Walk-forward only:
the model for year Y is fitted on data up to the end of year Y-1 (a 7-day gap so targets do not overlap), so
2023-2026 are out of sample. Fixed in advance:

  features  cross-sectional z-scores within the point-in-time top 30 each day (missing = 0): smart-money gap (3-day),
            taker buy/sell (3-day), open-interest growth (7-day), funding (3-day), past 7- and 30-day return,
            volume surge (7 vs 60 days), biggest daily jump (30 days), Coinbase premium (7-day), Korean volume share
  target    next 7-day return, demeaned across coins
  model     ridge regression (alpha = 10 on z-scores), refitted every January
  book      positioning_lab's 7-slice long 6 / short 6 with funding, on the model's score vs on smart money alone
  pass      the model's book beats smart money alone over 2023-2026 and in at least 3 of the 4 years

    python -m backend.combo_lab
"""

import os
import time

import numpy as np
import pandas as pd

from backend import positioning_lab as pl
from backend.universe_data import daily_panel

ALPHA = 10.0


def features():
    sig, ret, member, close = pl.signals()
    idx, cols = ret.index, ret.columns
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    qvol = daily_panel(last_month)["qvol"].reindex(index=idx, columns=cols)
    m = pl.metrics()
    oi = m.pivot_table(index="day", columns="sym", values="oi_value").reindex(index=idx, columns=cols)
    held = sorted(set(cols[member.any().to_numpy()]))
    fund = pl.daily_funding(held, idx).reindex(columns=cols).fillna(0.0)
    rc = close.pct_change(fill_method=None)
    f = {"smart": sig["smart"], "taker": sig["taker"], "oi": oi / oi.shift(7) - 1,
         "funding": fund.rolling(3, min_periods=2).mean(),
         "past7": (close / close.shift(7) - 1).reindex(idx), "past30": (close / close.shift(30) - 1).reindex(idx),
         "vol_surge": np.log(qvol.rolling(7, min_periods=5).mean() / qvol.shift(7).rolling(60, min_periods=40).mean()),
         "max_jump": rc.rolling(30, min_periods=20).max().reindex(idx)}
    # Coinbase premium and Korean share from their labs' caches (only coins covered by those venues)
    try:
        from backend import cb_premium_lab as cb
        prem = {}
        for s in cols:
            p1, p2 = os.path.join(cb.DATA, f"cb_{s[:-4]}.json"), os.path.join(cb.DATA, f"spot_{s}.csv")
            if os.path.exists(p1) and os.path.exists(p2):
                both = pd.DataFrame({"cb": cb.cb_closes(s[:-4]), "bn": cb.spot_closes(s)}).dropna()
                prem[s] = np.log(both["cb"] / both["bn"])
        f["cb_premium"] = pd.DataFrame(prem).reindex(index=idx, columns=cols).rolling(7, min_periods=5).mean()
    except Exception as e:                                  # noqa: BLE001
        print(f"  (no Coinbase premium: {e})")
    try:
        from backend import upbit_lab as ul
        sh = {}
        for s in cols:
            path = os.path.join(ul.DATA, f"KRW-{s[:-4].replace('1000', '')}.json")
            if os.path.exists(path):
                sh[s] = (ul.candles(f"KRW-{s[:-4].replace('1000', '')}") / qvol[s]).replace([np.inf, -np.inf], np.nan)
        f["korea_share"] = np.log(pd.DataFrame(sh).reindex(index=idx, columns=cols).rolling(7, min_periods=5).mean())
    except Exception as e:                                  # noqa: BLE001
        print(f"  (no Korean share: {e})")
    return f, ret, member, fund, close


def main():
    f, ret, member, fund, close = features()
    z = lambda x: x.where(member).sub(x.where(member).mean(axis=1), axis=0).div(x.where(member).std(axis=1), axis=0)
    Z = {k: z(v.reindex(index=ret.index, columns=ret.columns)).fillna(0.0).where(member) for k, v in f.items()}
    fwd = (close.shift(-7) / close - 1).reindex(index=ret.index, columns=ret.columns).where(member)
    fwd = fwd.sub(fwd.mean(axis=1), axis=0)
    names = list(Z)
    long = pd.concat({k: v.stack() for k, v in Z.items()}, axis=1)
    long["y"] = fwd.stack()
    long = long.dropna()
    dates = long.index.get_level_values(0)
    score = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    coefs = {}
    for year in (2023, 2024, 2025, 2026):
        train = long[dates < pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=7)]
        X, y = train[names].to_numpy(), train["y"].to_numpy()
        beta = np.linalg.solve(X.T @ X + ALPHA * np.eye(len(names)), X.T @ y)
        coefs[year] = dict(zip(names, beta))
        mask = (ret.index >= f"{year}-01-01") & (ret.index < f"{year + 1}-01-01")
        s = sum(Z[k].loc[mask] * b for k, b in zip(names, beta))
        score.loc[mask] = s
    print("ridge weights (x1000) fitted before each year:")
    print((pd.DataFrame(coefs) * 1000).round(2).to_string())

    def book(s):
        return pd.concat([pl.backtest(s, ret, member, offset=o, funding=fund) for o in range(7)], axis=1).mean(axis=1)
    model = book(score).loc["2023":]
    smart = book(f["smart"]).loc["2023":]
    print("\nout of sample, with funding:")
    for lab, r in (("model (all signals)", model), ("smart money alone", smart)):
        yearly = " ".join(f"{y}: {x.mean() / x.std() * np.sqrt(365):+.2f}" for y, x in r.groupby(r.index.year))
        eq = (1 + r).cumprod()
        print(f"  {lab:22s} 2023-now Sharpe {r.mean() / r.std() * np.sqrt(365):+.2f}, {eq.iloc[-1] ** (365 / len(r)) - 1:+.0%}/yr, "
              f"DD {(eq / eq.cummax() - 1).min():+.0%} | by year {yearly}")


if __name__ == "__main__":
    main()
