"""
The plan's history since 2015 with the hindsight taken out (follow-up to history_lab.py).

What history_lab.py knew in advance, and what this lab does instead:
  coins      today's 8 winners (SOL, AVAX, SUI picked in hindsight)
             -> each month, the 8 busiest coins by the past 30 days' dollar volume, among ALL coins Binance listed at
                the time, dead ones included (LUNA, FTT, ...). Binance spot before 2020-01, USD-M futures after.
                A coin that drops out of the 8 keeps its open trade until the rules exit; it gets no new entries.
  settings   120/60 bars and a 4x ATR stop, chosen on 2020-24 data
             -> walk-forward: every January from 2019 the setting with the best Sharpe on ALL EARLIER data is picked
                from 18 candidates (entry 60/120/180 bars x exit 30/60 x stop 3/4/5 ATR) and used for that year only.
                2017-18 use the textbook Turtle rule (20-day entry, 10-day exit = 120/60 4h bars) with a 4x ATR stop.
  funding    2025-26 funding rates for gold/silver/S&P/Nasdaq perps, which did not exist before 2026
             -> each asset pays max(its 2025-26 rate, 3-month US T-bill + 2%) a year on what it holds.

    python -m backend.honest_lab
"""

import io
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
import requests

from backend import tradfi_book as tb
from backend.cross_asset_lab import FUNDING_2026, etf_prices
from backend.growth_study import BULK_CACHE, LIVE, _bulk_csv, _rows, load_market_bulk
from backend.strategy_lab import trend_returns
from backend import strategy_lab
from backend.universe_data import S3, _s3_list, available_months, daily_panel

SPOT = "data/spot/monthly/klines"
TOP = 8
STABLE = {"USDC", "BUSD", "TUSD", "PAX", "USDS", "USDSB", "USDP", "FDUSD", "DAI", "EUR", "GBP", "AUD", "UST", "USDE",
          "SUSD", "BKRW", "IDRT", "BIDR", "AEUR", "ERD"}
GRID = [(e, x, s) for e in (60, 120, 180) for x in (30, 60) for s in (3.0, 4.0, 5.0)]
TEXTBOOK = (120, 60, 4.0)
FUNDING = dict(FUNDING_2026, GLD=0.04)
COST = 0.0005


# ---------- crypto data: spot before futures ----------
def spot_symbols() -> list:
    path = os.path.join(BULK_CACHE, "spot_usdt_symbols.txt")
    if os.path.exists(path):
        return open(path).read().split()
    syms = _s3_list(f"{SPOT}/", rf"<Prefix>{SPOT}/([^/]+)/</Prefix>")
    syms = [s for s in syms if s.endswith("USDT") and s[:-4] not in STABLE
            and not re.search(r"(UP|DOWN|BULL|BEAR)USDT$", s)]
    with open(path, "w") as f:
        f.write("\n".join(syms))
    return syms


def spot_months(sym: str, interval: str) -> list:
    path = os.path.join(BULK_CACHE, "months", f"spot-{sym}-{interval}.txt")
    if os.path.exists(path):
        return open(path).read().split()
    keys = _s3_list(f"{SPOT}/{sym}/{interval}/", rf"{sym}-{interval}-(\d{{4}}-\d{{2}})\.zip<")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(sorted(set(keys))))
    return sorted(set(keys))


def spot_csv(sym: str, interval: str, month: str) -> str:
    name = f"{sym}-{interval}-{month}"
    path = os.path.join(BULK_CACHE, "spot_" + name + ".csv")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    for attempt in range(5):
        try:
            r = requests.get(f"https://data.binance.vision/{SPOT}/{sym}/{interval}/{name}.zip", timeout=30)
            text = "" if r.status_code == 404 else zipfile.ZipFile(io.BytesIO(r.content)).read(name + ".csv").decode()
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def spot_daily(until: str = "2019-12") -> dict:
    """close / qvol (dates x symbols) from Binance spot, months up to `until`."""
    syms = spot_symbols()
    with ThreadPoolExecutor(32) as ex:
        months = dict(zip(syms, ex.map(lambda s: [m for m in spot_months(s, "1d") if m <= until], syms)))
    jobs = [(s, m) for s, ms in months.items() for m in ms]
    with ThreadPoolExecutor(32) as ex:
        texts = list(ex.map(lambda sm: spot_csv(sm[0], "1d", sm[1]), jobs))
    close, qvol = {}, {}
    for (s, _), text in zip(jobs, texts):
        for r in _rows(text):
            d = pd.Timestamp(int(r[0]), unit="ms")
            close.setdefault(s, {})[d] = float(r[4])
            qvol.setdefault(s, {})[d] = float(r[7])
    return {"close": pd.DataFrame(close).sort_index(), "qvol": pd.DataFrame(qvol).sort_index()}


def membership(spot: dict, fut: dict) -> pd.DataFrame:
    """Top TOP by trailing 30-day quote volume, re-picked on each month's first day (60+ days of history).
    Spot volumes before 2020-01-01, futures volumes after."""
    cut = pd.Timestamp("2020-01-01")
    close = pd.concat([spot["close"][:cut - pd.Timedelta(days=1)], fut["close"][cut:]]).sort_index()
    qvol = pd.concat([spot["qvol"][:cut - pd.Timedelta(days=1)], fut["qvol"][cut:]]).sort_index()
    close = close.groupby(level=0).last()
    qvol = qvol.groupby(level=0).last()
    vol = qvol.rolling(30, min_periods=30).mean().shift(1)
    history = close.notna().cumsum().shift(1) >= 60
    vol = vol.where(history & close.notna())
    first = vol.index.to_series().dt.is_month_start
    top = vol[first].rank(axis=1, ascending=False, method="first") <= TOP
    return top.reindex(vol.index).ffill().fillna(False).astype(bool)


def crypto_market(member: pd.DataFrame, last_month: str) -> dict:
    ever = [s for s in member.columns if member[s].any()]
    fut_syms = [s for s in ever if available_months(s, "4h")]
    jobs = [(s, m) for s in fut_syms for m in available_months(s, "4h") if m <= last_month]
    with ThreadPoolExecutor(24) as ex:
        list(ex.map(lambda sm: _bulk_csv("klines", sm[0], sm[1], "4h"), jobs))
    fut = load_market_bulk("4h", last_month, fut_syms, lambda s: available_months(s, "4h"))
    market = {}
    for sym in ever:
        f = fut.get(sym, {"bars": [], "funding": np.zeros(0)})
        first = f["bars"][0]["timestamp"] if f["bars"] else 10 ** 15
        first_month = pd.Timestamp(first, unit="ms").strftime("%Y-%m") if f["bars"] else last_month
        spot = []
        for m in spot_months(sym, "4h"):
            if m <= first_month:
                spot += [{"timestamp": int(r[0]), "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
                          "close": float(r[4]), "volume": float(r[5])} for r in _rows(spot_csv(sym, "4h", m))]
        spot = sorted({b["timestamp"]: b for b in spot if b["timestamp"] < first}.values(), key=lambda b: b["timestamp"])
        bars = spot + f["bars"]
        if len(bars) < 250:
            continue
        market[sym] = {"bars": bars, "funding": np.concatenate([np.zeros(len(spot)), f["funding"]]),
                       "idx": {b["timestamp"]: i for i, b in enumerate(bars)}}
    return market


def crypto_stream(market, member, e, x, s, level) -> pd.Series:
    p = replace(LIVE, entry_n=e, exit_n=x, stop_atr=s, risk_pct=tb.CRYPTO_RISK_PER_LEVEL * level)
    return trend_returns(market, p, member)


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(365)) if len(r) > 60 and r.std() > 0 else -np.inf


def walk_forward(streams: dict, years) -> dict:
    """year -> grid setting chosen on all data before that year."""
    pick = {}
    for y in years:
        if y < 2019:
            pick[y] = TEXTBOOK
        else:
            pick[y] = max(GRID, key=lambda g: sharpe(streams[g][:f"{y - 1}-12-31"]))
    return pick


# ---------- TradFi with no-hindsight funding ----------
def tbill() -> pd.Series:
    import yfinance as yf
    r = yf.download("^IRX", start="2013-01-01", progress=False, auto_adjust=True)["Close"]
    r = r.iloc[:, 0] if isinstance(r, pd.DataFrame) else r
    return (r / 100).dropna()


def tradfi_returns(px: pd.DataFrame, level: float, rate: pd.Series) -> pd.Series:
    px = px[sorted(set(tb.ASSETS.values()))]
    ret = px.pct_change(fill_method=None)
    month_end = px.index.to_series().groupby(px.index.to_period("M")).transform("max") == px.index.to_series()
    rows = {}
    for d in px.index[month_end.to_numpy()]:
        h = px.loc[:d].dropna()
        if len(h) < tb.MOM_DAYS + 1:
            continue
        rows[d] = {tb.ASSETS[k]: v for k, v in tb.target_weights(tb.signals(h), tb.TRADFI_VOL_PER_LEVEL * level).items()}
    w = pd.DataFrame(rows).T.reindex(px.index).ffill().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    rf = rate.reindex(px.index).ffill().bfill()
    fund = pd.DataFrame({c: np.maximum(FUNDING.get(c, 0.0), rf + 0.02) for c in held.columns}, index=px.index)
    cost = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1) * COST + (held * fund / 252).sum(axis=1)
    return (held * ret.fillna(0.0)).sum(axis=1) - cost


def deposit_paths(r: pd.Series, deposit: float = 300.0, months: int = 24) -> pd.Series:
    """Real history: EUR 500 + deposit every 30 days, started on each month's first day; value after `months`."""
    out = {}
    eq_all = r.to_numpy()
    idx = r.index
    for start in pd.date_range(idx[0], idx[-1] - pd.Timedelta(days=30 * months), freq="MS"):
        i0 = idx.searchsorted(start)
        eq = 500.0
        for m in range(months):
            eq = eq * np.prod(1 + eq_all[i0 + 30 * m:i0 + 30 * (m + 1)]) + deposit
        out[start] = eq
    return pd.Series(out)


def main():
    t0 = time.time()
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    spot = spot_daily()
    fut = daily_panel(last_month)
    member = membership(spot, fut)
    market = crypto_market(member, last_month)
    member = member[[s for s in member.columns if s in market]]
    print(f"{len(market)} coins were ever in the monthly top {TOP} ({time.time() - t0:.0f}s)")
    for y in range(2017, int(last_month[:4]) + 1):
        picks = member[str(y)].any()
        print(f"  {y}: " + " ".join(s.replace("USDT", "") for s in picks[picks].index))

    strategy_lab.START = "2017-09-01"
    base = {g: crypto_stream(market, member, *g, 1) for g in GRID}
    print(f"grid run ({time.time() - t0:.0f}s)")
    years = list(range(2015, int(last_month[:4]) + 1))
    pick = walk_forward(base, years)
    print("walk-forward settings (entry/exit bars, stop ATR): " +
          ", ".join(f"{y}: {e}/{x}/{s:.0f}" for y, (e, x, s) in pick.items() if y >= 2017))
    px = etf_prices("2013-06-01")
    rate = tbill()

    for level in (1, 2, 3):
        need = {g for y, g in pick.items() if y >= 2017}
        streams = {g: crypto_stream(market, member, *g, level) for g in need}
        live = crypto_stream(market, member, *TEXTBOOK, level)
        days = pd.date_range("2015-01-01", max(s.index.max() for s in streams.values()), freq="D")
        c = pd.Series(0.0, index=days)
        for y in years:
            if y >= 2017:
                s = streams[pick[y]].reindex(days).fillna(0.0)
                c[str(y)] = s[str(y)]
        live = live.reindex(days).fillna(0.0)
        t = tradfi_returns(px, level, rate).reindex(days).fillna(0.0)
        plan = c + t
        print(f"\nLEVEL {level}  year | crypto top-{TOP} walk-fwd | (live setting) | tradfi | PLAN | worst drop | EUR 500 since 2015")
        eq = 500.0
        for y in years:
            e = (1 + plan[str(y)]).cumprod()
            eq *= e.iloc[-1]
            print(f"         {y}{'*' if y == years[-1] else ' '} | {(1 + c[str(y)]).prod() - 1:+7.0%} | {(1 + live[str(y)]).prod() - 1:+7.0%} | "
                  f"{(1 + t[str(y)]).prod() - 1:+6.0%} | {e.iloc[-1] - 1:+6.0%} | {(e / e.cummax().clip(lower=1) - 1).min():+5.0%} | {eq:,.0f}")
        e = (1 + plan).cumprod()
        print(f"         whole period: {e.iloc[-1] ** (365 / len(plan)) - 1:+.0%}/yr, max drawdown {(e / e.cummax() - 1).min():+.0%}")
        dp = deposit_paths(plan["2017-10-01":])
        print(f"         EUR 500 + 300/month (7,700 paid in), after 24 months, every start month 2017-10..: median {dp.median():,.0f}, "
              f"worst {dp.min():,.0f}, best {dp.max():,.0f}, below paid-in {np.mean(dp < 7700):.0%}, >= 10k {np.mean(dp >= 10_000):.0%}")


if __name__ == "__main__":
    main()
