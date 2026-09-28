"""
A disciplined large search for "the 1 in a million" cross-sectional crypto signal, with a locked holdout.

Searching many ideas always finds some that look great by luck; the protection is to (1) fix the search space, (2)
choose winners on one period only, (3) measure how good the best of many RANDOM signals looks on the same period, and
(4) give only the top few one look at data kept locked away.

  data      point-in-time top 30 Binance USDT perps (positioning_lab), 2022-01 .. now: top-trader position ratio,
            all-account ratio, taker buy/sell, open interest, funding, price, quote volume
  space     8 base series x 9 transforms (mean over 1/3/7/14 days; change of the 3/7-day mean vs the prior 30/60;
            z-score of the 3/7-day mean vs its own past 90 days) x both signs x holding 1/3/7/14 days (as many daily
            slices) x 4/6/10 coins per leg = 1,728 variants
  book      long the top k / short the bottom k, equal weight, 0.1% per unit of turnover, real funding
  search    2022-01-01 .. 2025-06-30; a variant must also be positive in both halves of that period
  null      the same selection over 200 random signals (noise) gives the Sharpe the best of many looks like by luck
  holdout   2025-07-01 .. now, opened once for the top 5 variants (one per base series)

    python -m backend.search_lab
"""

import itertools
import time

import numpy as np
import pandas as pd

from backend import positioning_lab as pl
from backend.universe_data import daily_panel

SEARCH_END, HOLD_START = "2025-06-30", "2025-07-01"
COST = 0.001


def book_returns(S: np.ndarray, M: np.ndarray, R: np.ndarray, F: np.ndarray, k: int, h: int) -> np.ndarray:
    """Daily returns of the staggered long/short book (vectorised): held portfolio = mean of the last h daily
    portfolios; R, F are next-day returns / funding aligned with the formation day."""
    X = np.where(M & np.isfinite(S), S, np.nan)
    n = np.isfinite(X).sum(axis=1)
    lo = np.argsort(np.where(np.isfinite(X), X, np.inf), axis=1)[:, :k]
    hi = np.argsort(np.where(np.isfinite(X), -X, np.inf), axis=1)[:, :k]
    W = np.zeros_like(R)
    rows = np.arange(len(X))[:, None]
    ok = (n >= 2 * k)[:, None]
    W[rows, hi] += np.where(ok, 0.5 / k, 0.0)
    W[rows, lo] -= np.where(ok, 0.5 / k, 0.0)
    if h > 1:
        c = np.cumsum(W, axis=0)
        H = (c - np.vstack([np.zeros((h, W.shape[1])), c[:-h]])) / h
    else:
        H = W
    turn = np.abs(np.diff(np.vstack([np.zeros((1, H.shape[1])), H]), axis=0)).sum(axis=1)
    return (H * R).sum(axis=1) - COST * turn - (H * F).sum(axis=1)


def sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) > 30 and x.std() > 0 else np.nan


def transforms(base: pd.DataFrame) -> dict:
    out = {}
    for L in (1, 3, 7, 14):
        out[f"mean{L}"] = base.rolling(L, min_periods=max(1, L // 2)).mean()
    for L, P in ((3, 30), (7, 30), (7, 60)):
        out[f"chg{L}v{P}"] = base.rolling(L, min_periods=max(1, L // 2)).mean() - base.shift(L).rolling(P, min_periods=P // 2).mean()
    for L in (3, 7):
        m = base.rolling(L, min_periods=max(1, L // 2)).mean()
        out[f"z{L}"] = (m - m.shift(L).rolling(90, min_periods=45).mean()) / m.shift(L).rolling(90, min_periods=45).std()
    return out


def main():
    t0 = time.time()
    sig, ret, member, close = pl.signals()
    idx, cols = ret.index, ret.columns
    m = pl.metrics()
    wide = {c: m.pivot_table(index="day", columns="sym", values=c).reindex(index=idx, columns=cols)
            for c in ("top_ls", "acct_ls", "taker_ls", "oi_value")}
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    qvol = daily_panel(last_month)["qvol"].reindex(index=idx, columns=cols)
    held = sorted(set(cols[member.any().to_numpy()]))
    fund = pl.daily_funding(held, idx).reindex(columns=cols).fillna(0.0)
    bases = {"smart_gap": np.log(wide["top_ls"]) - np.log(wide["acct_ls"]), "top_ratio": np.log(wide["top_ls"]),
             "all_accounts": np.log(wide["acct_ls"]), "taker": np.log(wide["taker_ls"]), "open_interest": np.log(wide["oi_value"]),
             "funding": fund.where(member), "price": np.log(close.reindex(index=idx, columns=cols)), "volume": np.log(qvol)}
    M = member.to_numpy(bool)
    R = ret.shift(-1).fillna(0.0).to_numpy()
    F = fund.shift(-1).fillna(0.0).to_numpy()
    search = np.asarray(idx <= SEARCH_END)
    half = np.asarray(idx <= pd.Timestamp("2023-09-30"))
    hold = np.asarray(idx >= HOLD_START)
    rows = []
    for bname, base in bases.items():
        for tname, S in transforms(base.replace([np.inf, -np.inf], np.nan)).items():
            Sv = S.to_numpy()
            for sign, h, k in itertools.product((1, -1), (1, 3, 7, 14), (4, 6, 10)):
                r = book_returns(sign * Sv, M, R, F, k, h)
                rows.append({"base": bname, "transform": tname, "sign": sign, "hold": h, "legs": k,
                             "search": sharpe(r[search]), "h1": sharpe(r[search & half]), "h2": sharpe(r[search & ~half]),
                             "holdout": sharpe(r[hold]), "r": r})
    res = pd.DataFrame(rows)
    print(f"{len(res)} variants in {time.time() - t0:.0f}s; search 2022-01 .. {SEARCH_END}, holdout {HOLD_START} .. now")

    # null: the best search-period Sharpe among random signals, same selection (both halves positive)
    rng = np.random.default_rng(0)
    null_best = []
    for _ in range(200):
        noise = rng.standard_normal(R.shape)
        noise = pd.DataFrame(noise).rolling(7, min_periods=1).mean().to_numpy()      # as persistent as real signals
        best = -np.inf
        for sign, h, k in itertools.product((1, -1), (1, 7), (6,)):
            r = book_returns(sign * noise, M, R, F, k, h)
            if sharpe(r[search & half]) > 0 and sharpe(r[search & ~half]) > 0:
                best = max(best, sharpe(r[search]))
        null_best.append(best)
    null_best = np.array(null_best)
    null_best = null_best[np.isfinite(null_best)]
    n_null_trials = 4 * len(null_best)
    print(f"random signals: best of 4 variants each -> search Sharpe median {np.median(null_best):.2f}, 95th pct "
          f"{np.percentile(null_best, 95):.2f}; scaled to {len(res)} variants the luck level is roughly "
          f"{np.percentile(null_best, 95) * np.sqrt(np.log(len(res)) / np.log(4)):.2f}")

    ok = res[(res["h1"] > 0) & (res["h2"] > 0)].sort_values("search", ascending=False)
    print(f"\n{len(ok)} variants positive in both halves of the search period. Top 15 by search Sharpe:")
    print(ok.head(15)[["base", "transform", "sign", "hold", "legs", "search", "h1", "h2"]].round(2).to_string(index=False))
    top = ok.drop_duplicates("base").head(5)
    print("\nHOLDOUT (opened once) for the best variant of each of the top 5 base series:")
    print(top[["base", "transform", "sign", "hold", "legs", "search", "holdout"]].round(2).to_string(index=False))
    print("\nby base series: best search Sharpe (all variants) and share of variants positive in both halves")
    print(res.groupby("base").apply(lambda g: pd.Series({"best_search": g["search"].max(),
                                                          "share_ok": ((g["h1"] > 0) & (g["h2"] > 0)).mean()})).round(2).to_string())


if __name__ == "__main__":
    main()
