"""
Strategy lab: can a wider coin universe or extra, less correlated strategies raise the Sharpe ratio
enough to matter for the $500 -> $10k goal?

Why Sharpe: at the growth-optimal (Kelly) leverage, a strategy with Sharpe S grows the median account
by about exp(S^2 / 2) per year. The live trend strategy (Sharpe ~1.2 out of sample) tops out near 2x a
year; 20x in 12 months needs S ~2.5, in 6 months S ~3.5. Past Kelly, more leverage lowers growth.

Everything uses a point-in-time universe (top N USDT perps by trailing 30-day volume, delisted coins
included, see universe_data.py), taker fees + slippage on every trade and real funding.

  A. TREND-8     the live strategy on its 8 coins (baseline)
  B. TREND-N     the same rules on the top-N universe (entries only while a coin is in the top N)
  C. XSMOM       cross-sectional momentum: weekly, long the strongest / short the weakest quintile
  D. CARRY       funding carry: short the perp + hold spot on the coins paying the highest funding

Parameters for C and D are picked on 2020-06 .. 2024-06 only; 2024-07 .. now is reported separately.

    python -m backend.strategy_lab
"""

import time
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backend.backtest_trend import SPLIT, UNIVERSE
from backend.growth_study import LIVE, load_market_bulk, run_account
from backend.trend_strategy import compute_features
from backend.universe_data import available_months, daily_panel, load_funding, top_by_volume

START = "2020-06-01"
TOP_N = 30
FEE, SLIP = 0.0005, 0.0005          # perp taker fee and slippage per side
SPOT_FEE = 0.001                     # spot taker fee (carry's hedge leg)
YEAR = 365


# ---------- stats ----------
def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    eq = (1 + r).cumprod()
    years = len(r) / YEAR
    return {"cagr": eq.iloc[-1] ** (1 / years) - 1, "vol": r.std() * np.sqrt(YEAR),
            "sharpe": r.mean() / r.std() * np.sqrt(YEAR) if r.std() > 0 else np.nan,
            "maxdd": (eq / eq.cummax() - 1).min()}


def fmt(name: str, r: pd.Series) -> str:
    out = [f"{name:<34}"]
    for lo, hi in ((START, SPLIT), (SPLIT, None)):
        s = stats(r[lo:hi] if hi else r[lo:])
        out.append(f"CAGR {s['cagr']:+7.1%} vol {s['vol']:5.1%} Sharpe {s['sharpe']:+5.2f} DD {s['maxdd']:+6.1%}")
    return " | ".join(out)


# ---------- A/B: trend ----------
def trend_returns(market: Dict[str, dict], p, member: pd.DataFrame = None) -> pd.Series:
    feats = {s: compute_features(m["bars"], p) for s, m in market.items()}
    timeline = sorted({b["timestamp"] for m in market.values() for b in m["bars"]})
    can_enter = None
    if member is not None:
        days = {d.value // 10**6: row for d, row in zip(member.index, member.to_numpy())}
        cols = {s: k for k, s in enumerate(member.columns)}

        def can_enter(sym, ts):
            row = days.get(ts // 86_400_000 * 86_400_000)
            return row is not None and sym in cols and bool(row[cols[sym]])
    curve: List[Tuple[int, float]] = []
    start_ms = int(pd.Timestamp(START).value // 10**6) - 30 * 86_400_000
    run_account(market, feats, timeline, p, start_ms, timeline[-1] + 1, liquidation=False,
                can_enter=can_enter, curve=curve)
    eq = pd.Series(dict(curve))
    eq.index = pd.to_datetime(eq.index, unit="ms")
    daily = eq.resample("1D").last().dropna()
    return daily.pct_change().dropna()


# ---------- C: cross-sectional momentum ----------
def xsmom_returns(close: pd.DataFrame, funding: pd.DataFrame, member: pd.DataFrame,
                  lookback: int, hold: int = 7, frac: float = 0.2) -> pd.Series:
    ret = close.pct_change(fill_method=None)
    mom = close / close.shift(lookback) - 1
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    cur = pd.Series(0.0, index=close.columns)
    for k, d in enumerate(close.index):
        if k % hold == 0:
            m = mom.loc[d].where(member.loc[d]).dropna()
            cur = pd.Series(0.0, index=close.columns)
            n = int(len(m) * frac)
            if n >= 2:
                ranked = m.sort_values()
                cur[ranked.index[-n:]] = 0.5 / n
                cur[ranked.index[:n]] = -0.5 / n
        w.loc[d] = cur
    held = w.shift(1).fillna(0.0)                              # decided at close d, earns day d+1
    gross = (held * ret.fillna(0.0)).sum(axis=1)
    fund = (held * funding.reindex_like(held).fillna(0.0)).sum(axis=1)   # longs pay, shorts receive
    cost = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1) * (FEE + SLIP)
    return (gross - fund - cost)[START:]


# ---------- D: funding carry ----------
def carry_returns(funding: pd.DataFrame, member: pd.DataFrame, top_k: int = 5, hold: int = 7,
                  min_daily: float = 0.0003) -> pd.Series:
    """Short perp + long spot, equal notional over the top_k members by trailing 7-day funding.
    Returns per unit of capital (spot leg fully paid, perp margined at 3x -> 1.33 capital per notional).
    Ignores basis moves between spot and perp."""
    trail = funding.rolling(7, min_periods=7).mean()
    member = member.reindex(index=funding.index, columns=funding.columns).fillna(False).astype(bool)
    w = pd.DataFrame(0.0, index=funding.index, columns=funding.columns)
    cur = pd.Series(0.0, index=funding.columns)
    for k, d in enumerate(funding.index):
        if k % hold == 0:
            f = trail.loc[d].where(member.loc[d]).dropna()
            f = f[f > min_daily].sort_values()
            cur = pd.Series(0.0, index=funding.columns)
            if len(f):
                cur[f.index[-top_k:]] = 1.0 / min(top_k, len(f))
        w.loc[d] = cur
    held = w.shift(1).fillna(0.0)
    earned = (held * funding.fillna(0.0)).sum(axis=1)
    cost = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1) * (FEE + SPOT_FEE + 2 * SLIP)
    return ((earned - cost) / (1 + 1 / 3))[START:]


# ---------- growth odds from a daily return stream ----------
def odds(r: pd.Series, leverages, horizon_days: int = 365, step: int = 14, target: float = 20.0) -> pd.DataFrame:
    """Rolling fresh accounts compounding L x the daily returns: P(20x within 6/12 months), P(-90%)."""
    x = r.to_numpy()
    rows = []
    for L in leverages:
        h6 = h12 = ruin = 0
        med = []
        starts = range(0, len(x) - horizon_days, step)
        for s in starts:
            path = np.cumprod(np.maximum(1 + L * x[s:s + horizon_days], 0.0))
            hit = np.flatnonzero(path >= target)
            dead = np.flatnonzero(path <= 0.1)
            first_hit = hit[0] if len(hit) else None
            if first_hit is not None and (not len(dead) or first_hit < dead[0]):
                h12 += 1
                h6 += first_hit < 182
            elif len(dead):
                ruin += 1
            med.append(path[-1])
        n = len(starts)
        rows.append({"lev": L, "20x<=6mo": h6 / n, "20x<=12mo": h12 / n, "-90%": ruin / n,
                     "median 12mo": float(np.median(med))})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    panel = daily_panel(last_month)
    close = panel["close"]
    member = top_by_volume(panel["qvol"], close, TOP_N)
    ever = [s for s in member.columns if member[s].any()]
    print(f"{close.shape[1]} perps, {len(ever)} ever in the top {TOP_N} ({time.time() - t0:.0f}s)", flush=True)

    fund = {}
    for s in ever:
        f = load_funding(s, available_months(s, "1d"))
        fund[s] = f.resample("1D").sum() if len(f) else pd.Series(dtype=float)
    funding = pd.DataFrame(fund).reindex(close.index).fillna(0.0)
    close_n, member_n = close[ever], member[ever]
    print(f"funding loaded ({time.time() - t0:.0f}s)", flush=True)

    streams: Dict[str, pd.Series] = {}
    print("\nIn-sample 2020-06 .. 2024-06 | out-of-sample 2024-07 .. now")

    m8 = load_market_bulk("4h", last_month, UNIVERSE, lambda s: available_months(s, "4h"))
    streams["TREND-8 (live, 1% risk)"] = trend_returns(m8, LIVE)
    print(fmt("TREND-8 (live, 1% risk)", streams["TREND-8 (live, 1% risk)"]), flush=True)

    mN = load_market_bulk("4h", last_month, ever, lambda s: available_months(s, "4h"))
    for risk in (0.01, 0.005):
        name = f"TREND-{TOP_N} ({risk:.1%} risk)"
        p = replace(LIVE, risk_pct=risk, max_gross_leverage=6.0)
        streams[name] = trend_returns(mN, p, member_n)
        print(fmt(name, streams[name]), flush=True)

    print("\nXSMOM grid (lookback days / quintile), in-sample Sharpe picks the setting:")
    best, best_is = None, -9
    for lb in (7, 14, 28, 56):
        for frac in (0.2, 0.33):
            r = xsmom_returns(close_n, funding, member_n, lb, 7, frac)
            s_is = stats(r[START:SPLIT])["sharpe"]
            print("  " + fmt(f"XSMOM lb={lb} frac={frac}", r), flush=True)
            if s_is > best_is:
                best, best_is = (lb, frac, r), s_is
    streams[f"XSMOM lb={best[0]} frac={best[1]}"] = best[2]

    print("\nCARRY grid (top k / hold days):")
    best, best_is = None, -9
    for k in (3, 5, 10):
        for hold in (1, 7):
            r = carry_returns(funding, member_n, k, hold)
            s_is = stats(r[START:SPLIT])["sharpe"]
            print("  " + fmt(f"CARRY k={k} hold={hold}", r), flush=True)
            if s_is > best_is:
                best, best_is = (k, hold, r), s_is
    streams[f"CARRY k={best[0]} hold={best[1]}"] = best[2]

    df = pd.DataFrame(streams).dropna()
    print("\nCorrelation of daily returns:")
    print(df.corr().round(2).to_string())

    # Combine: weights = inverse in-sample volatility (equal risk), then scaled to 1% daily-vol-ish units
    names = [n for n in df.columns if not n.startswith("TREND-8") and "0.5%" not in n]
    vol_is = df[names][START:SPLIT].std()
    w = (1 / vol_is) / (1 / vol_is).sum()
    combo = (df[names] * w).sum(axis=1)
    combo = combo * (df["TREND-8 (live, 1% risk)"][START:SPLIT].std() / combo[START:SPLIT].std())
    print("\nWeights (equal risk, in-sample vol): " + ", ".join(f"{n} {x:.0%}" for n, x in w.items()))
    print(fmt("COMBINED (same vol as TREND-8)", combo))

    print("\nGrowth odds, fresh account every 2 weeks, 12-month runs, L x the daily returns:")
    for name, r in (("TREND-8", df["TREND-8 (live, 1% risk)"]), ("COMBINED", combo)):
        print(f"  {name}")
        print(odds(r, (1, 2, 3, 5, 7.5, 10)).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
