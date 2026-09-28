"""
Big traders vs everyone, across coins. Binance publishes, per perp, the long/short ratio of its top traders (by
position size) and of all accounts. When the top traders lean more long than the crowd on one coin than on others,
do they know something? insight_lab.py used these ratios as a filter on the 8 trend coins over time (they flipped
sign); this is the cross-sectional, market-neutral version on many coins. Fixed in advance:

  universe   point-in-time top 30 USDT perps by 30-day volume (delisted included), 2022 .. now
  signals    smart  = 3-day mean of log(top-trader L/S by position) - log(all-account L/S)
             taker  = 3-day mean of log(taker buy / sell volume)
             oi     = 7-day change in open interest (value)
             (only the days each formation needs are downloaded: d, d-1, d-2 and d-7 for coins in the top 30)
  rule       every 7 days, long the 6 coins with the highest signal and short the 6 lowest, equal weight, hold 7
             days; 0.1% per unit of turnover; funding ignored (both legs pay / receive similar amounts on average)
  pass       Sharpe > 0.5 in both 2022-23 and 2024+ with the same sign

    python -m backend.positioning_lab collect        # add collect_full for every weekday
    python -m backend.positioning_lab
    python -m backend.positioning_lab robust
"""

import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from backend.universe_data import daily_panel, top_by_volume

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "positioning"))

TOP_N, LEGS, COST, SPLIT = 30, 6, 0.001, "2024-01-01"


def universe():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    panel = daily_panel(last_month)
    top = top_by_volume(panel["qvol"], panel["close"], TOP_N).loc["2021-12-01":]
    return panel["close"], top


def formations(idx) -> list:
    return list(idx[::7])


def needed(close, top, full: bool = False) -> dict:
    """Days to download: d, d-1, d-2, d-7 for each formation day d (full: every day d-9 .. d for any member, so
    other rebalance weekdays can be tested)."""
    idx = close.loc["2022-01-01":].index
    need = {}
    for d in (idx if full else formations(idx)):
        for s in top.columns[top.loc[d].fillna(False).to_numpy(bool)]:
            need.setdefault(s, set()).update(d - pd.Timedelta(days=k) for k in ((0, 1, 2, 7) if not full else range(10)))
    return need


def _one(sym: str, day: pd.Timestamp):
    url = f"https://data.binance.vision/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return {"sym": sym, "day": day}
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
            num = df.drop(columns=["create_time", "symbol"]).apply(pd.to_numeric, errors="coerce")
            oi = num["sum_open_interest_value"].dropna()
            return {"sym": sym, "day": day, "oi_value": oi.iloc[-1] if len(oi) else np.nan,
                    "top_ls": num["sum_toptrader_long_short_ratio"].mean(),
                    "acct_ls": num["count_long_short_ratio"].mean(),
                    "taker_ls": num["sum_taker_long_short_vol_ratio"].mean()}
        except Exception:                                  # noqa: BLE001 - retried, then recorded as missing
            time.sleep(2 * (attempt + 1))
    return None


def metrics() -> pd.DataFrame:
    path = os.path.join(CACHE, "metrics.csv")
    return pd.read_csv(path, parse_dates=["day"]) if os.path.exists(path) else pd.DataFrame(columns=["sym", "day"])


def collect(full: bool = False):
    os.makedirs(CACHE, exist_ok=True)
    close, top = universe()
    need = needed(close, top, full)
    have = metrics()
    done = set(zip(have["sym"], pd.to_datetime(have["day"])))
    todo = [(s, d) for s, ds in need.items() for d in sorted(ds) if (s, d) not in done]
    print(f"{len(need)} symbols, {sum(len(v) for v in need.values())} symbol-days needed, {len(todo)} to fetch", flush=True)
    for i in range(0, len(todo), 2000):
        with ThreadPoolExecutor(32) as ex:
            rows = [r for r in ex.map(lambda a: _one(*a), todo[i:i + 2000]) if r]
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(os.path.join(CACHE, "metrics.csv"), index=False)
        print(f"  {min(i + 2000, len(todo))}/{len(todo)}", flush=True)


def signals():
    close, top = universe()
    idx = close.loc["2022-01-01":].index
    m = metrics()
    wide = {c: m.pivot_table(index="day", columns="sym", values=c).reindex(pd.date_range(idx[0] - pd.Timedelta(days=7), idx[-1]))
            for c in ("top_ls", "acct_ls", "taker_ls", "oi_value")}
    gap = np.log(wide["top_ls"]) - np.log(wide["acct_ls"])
    sig = {"smart": gap.rolling(3, min_periods=2).mean().reindex(idx),
           "smart_1d": gap.reindex(idx),
           "taker": np.log(wide["taker_ls"]).rolling(3, min_periods=2).mean().reindex(idx),
           "oi": (wide["oi_value"] / wide["oi_value"].shift(7) - 1).reindex(idx)}
    cols = sig["smart"].columns
    ret = close[cols].pct_change(fill_method=None).loc[idx]
    member = top[cols].reindex(idx).fillna(False)
    return sig, ret, member, close[cols]


def backtest(sig, ret, member, legs=LEGS, lag=0, offset=0, funding=None, side="both"):
    """Daily returns of the weekly long/short book. lag: extra days between the signal and the trade.
    funding: dates x symbols daily funding paid by longs (shorts receive it). side: both | long | short (a leg minus
    the equal-weight average of the eligible coins)."""
    idx = ret.index
    sig = sig.shift(lag)
    nxt = ret.shift(-1).fillna(0.0)
    rows, w_prev = [], pd.Series(0.0, index=ret.columns)
    for k, d in enumerate(idx[:-1]):
        if k % 7 == offset:
            s = sig.loc[d].where(member.loc[d]).dropna()
            w = pd.Series(0.0, index=ret.columns)
            if len(s) >= 2 * legs:
                ranked = s.sort_values()
                if side == "both":
                    w[ranked.index[-legs:]] += 1 / legs / 2
                    w[ranked.index[:legs]] -= 1 / legs / 2
                else:
                    w[ranked.index[-legs:] if side == "long" else ranked.index[:legs]] += 0.5 / legs
                    w[s.index] -= 0.5 / len(s)
            turn = (w - w_prev).abs().sum()
            w_prev = w
        else:
            turn = 0.0
        r = (w_prev * nxt.loc[d]).sum() - turn * COST
        if funding is not None:
            r -= (w_prev * funding.shift(-1).loc[d].reindex(w_prev.index).fillna(0.0)).sum()
        rows.append((d, r))
    return pd.Series(dict(rows))


def _fmt(r: pd.Series) -> str:
    cells = []
    for lab, x in (("2022-23", r[:SPLIT]), ("2024+", r[SPLIT:])):
        eq = (1 + x).cumprod()
        cells.append(f"{lab} {eq.iloc[-1] ** (365 / len(x)) - 1:+6.1%}/yr Sharpe {x.mean() / x.std() * np.sqrt(365):+.2f} "
                     f"DD {(eq / eq.cummax() - 1).min():+.0%}")
    return " | ".join(cells)


def main():
    sig, ret, member, _ = signals()
    f = formations(ret.index)
    print(f"{ret.shape[1]} coins; formation-day coverage of the top {TOP_N}: "
          f"{sig['smart'].loc[f].where(member.loc[f]).notna().sum().sum() / member.loc[f].sum().sum():.0%}")
    for name, key in (("smart money (top traders vs all)", "smart"), ("taker buying", "taker"),
                      ("open-interest growth", "oi")):
        print(f"  {name:34s} " + _fmt(backtest(sig[key], ret, member)))


def daily_funding(cols, idx) -> pd.DataFrame:
    from backend.universe_data import load_funding
    months = sorted({d.strftime("%Y-%m") for d in idx})
    out = {}
    with ThreadPoolExecutor(16) as ex:
        for s, f in zip(cols, ex.map(lambda s: load_funding(s, months), cols)):
            if len(f):
                out[s] = f.resample("1D").sum()
    return pd.DataFrame(out).reindex(idx).fillna(0.0)


def robust():
    sig, ret, member, close = signals()
    smart = sig["smart"]
    base = backtest(smart, ret, member)
    print("smart-money book, robustness (in-sample 2022-23 | 2024+):")
    print(f"  {'base (as pre-registered)':34s} " + _fmt(base))
    held = sorted(set(ret.columns[(member.any()).to_numpy()]))
    fund = daily_funding(held, ret.index).reindex(columns=ret.columns).fillna(0.0)
    print(f"  {'with funding':34s} " + _fmt(backtest(smart, ret, member, funding=fund)))
    print(f"  {'signal 1 day older':34s} " + _fmt(backtest(smart, ret, member, lag=1)))
    print(f"  {'1-day signal (no 3-day mean)':34s} " + _fmt(backtest(sig['smart_1d'], ret, member)))
    for legs in (4, 10):
        print(f"  {f'{legs} coins per leg':34s} " + _fmt(backtest(smart, ret, member, legs=legs)))
    for off in range(1, 7):
        print(f"  {f'rebalance day offset {off}':34s} " + _fmt(backtest(smart, ret, member, offset=off)))
    print(f"  {'long leg vs average':34s} " + _fmt(backtest(smart, ret, member, side='long')))
    print(f"  {'short leg vs average':34s} " + _fmt(backtest(smart, ret, member, side='short')))
    # is it just short-term reversal / momentum? rank on the past 7-day return instead, and on the residual
    past7 = close / close.shift(7) - 1
    print(f"  {'past 7-day return (momentum)':34s} " + _fmt(backtest(past7.reindex(ret.index), ret, member)))
    z = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
    zs, zp = z(smart.where(member)), z(past7.reindex(ret.index).where(member))
    beta = (zs * zp).sum(axis=1) / (zp * zp).sum(axis=1)
    resid = zs - zp.mul(beta, axis=0)
    print(f"  {'smart, momentum removed':34s} " + _fmt(backtest(resid, ret, member)))
    print(f"  cross-sectional correlation of smart with the past 7-day return: {zs.corrwith(zp, axis=1).mean():+.2f}")
    yearly = (1 + base).groupby(base.index.year).prod() - 1
    print("  by year: " + ", ".join(f"{y} {v:+.0%}" for y, v in yearly.items()))
    from backend.cross_asset_lab import BINANCE_TRADFI, FUNDING_2026, crypto_trend_returns, etf_prices, tsmom_returns
    from backend.strategy_lab import START, fmt
    crypto = crypto_trend_returns()
    days = pd.date_range(crypto.index.min(), crypto.index.max(), freq="D")
    cd = crypto.reindex(days).fillna(0.0)
    tb = tsmom_returns(etf_prices("2005-01-01")[list(BINANCE_TRADFI)], FUNDING_2026).reindex(days).fillna(0.0)
    pb = backtest(smart, ret, member, funding=fund).reindex(days).fillna(0.0)
    print(f"  correlation with the crypto trend book {pb['2022':].corr(cd['2022':]):+.2f}, with TradFi {pb['2022':].corr(tb['2022':]):+.2f}")
    print("\nPlan with the smart-money book (weights from 2022-01 .. 2024-06 volatility; the book starts in 2022):")
    for name, cols in (("plan (crypto + TradFi)", {"c": cd, "t": tb}), ("plan + smart-money book", {"c": cd, "t": tb, "p": pb})):
        both = pd.DataFrame(cols)["2022-01-01":]
        vol_is = both[:"2024-06-30"].std()
        w = (1 / vol_is) / (1 / vol_is).sum()
        combo = (both * w).sum(axis=1)
        combo *= cd["2022":"2024-06-30"].std() / combo[:"2024-06-30"].std()
        print(fmt(name, combo))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"collect": collect, "collect_full": lambda: collect(True), "main": main, "robust": robust}[cmd]()
