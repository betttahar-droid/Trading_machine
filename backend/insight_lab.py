"""
Do free market-insight sources tell which trend entries will work?

Sources (all free, all with multi-year history, all cached under data/insight_cache/):
  Binance futures metrics  data.binance.vision daily files (5-min rows): open interest, top-trader and
                           all-account long/short ratios, taker buy/sell volume ratio, per coin, 2021 on
  Fear & Greed index      api.alternative.me, 2018 on
  Stablecoin supply       DefiLlama (total USD-pegged circulating), 2017 on
  Deribit DVOL            BTC 30-day implied volatility index, 2021 on
  Coinbase premium        Coinbase BTC-USD vs Binance BTCUSDT daily close (US demand)
  Listing dilution        new USDT perps listed on Binance in the prior 30 days, and all active perps
                          (universe_data.py): more tokens competing for the same attention and money

For every entry of the live trend strategy (8 coins) each measure is taken from the day before entry.
Trades are split into thirds by the measure (top vs bottom third, permutation test), separately for
2020-03 .. 2024-06 and 2024-07 .. now. A measure is only interesting if the gap has the same sign and is
unlikely to be chance in both periods; one that flips sign would hurt as a filter.

    python -m backend.insight_lab
    python -m backend.insight_lab --composite

--composite is the portfolio-level test of a blend chosen before looking at 2024-07 on: the measures whose
2020-03 .. 2024-06 split looked useful (random split as large < 30%), each z-scored against its own trailing
year with the sign it had then, averaged per coin and day. The live 8-coin strategy skips entries when the
blend is in the bottom third of its trailing year; compared with no filter and with random skips.
"""

import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import numpy as np
import pandas as pd
import requests

from backend.backtest_trend import DATA_DIR, SPLIT, UNIVERSE, _ms
from backend.hype_lab import live_trend_trades, split_test
from backend.universe_data import _s3_list

CACHE = os.path.join(DATA_DIR, "insight_cache")
UA = {"User-Agent": "Mozilla/5.0 (research; github.com/betttahar-droid/Trading_machine)"}


def _cached_json(name: str, fetch, max_age: float = 86400):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age:
        return json.load(open(path))
    data = fetch()
    json.dump(data, open(path, "w"))
    return data


def binance_metrics(symbol: str) -> pd.DataFrame:
    """Daily: OI value at the day's last row, mean long/short ratios and taker ratio."""
    path = os.path.join(CACHE, f"metrics_{symbol}.csv")
    have = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else pd.DataFrame()
    prefix = f"data/futures/um/daily/metrics/{symbol}/"
    days = _s3_list(prefix, rf"{symbol}-metrics-(\d{{4}}-\d{{2}}-\d{{2}})\.zip<")
    todo = [d for d in sorted(set(days)) if pd.Timestamp(d) not in have.index]

    def one(day):
        for attempt in range(4):
            try:
                r = requests.get(f"https://data.binance.vision/{prefix}{symbol}-metrics-{day}.zip", timeout=30)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                z = zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
                num = df.drop(columns=["create_time", "symbol"]).apply(pd.to_numeric, errors="coerce")
                return {"day": pd.Timestamp(day), "oi_value": num["sum_open_interest_value"].dropna().iloc[-1]
                        if num["sum_open_interest_value"].notna().any() else np.nan,
                        "top_ls": num["sum_toptrader_long_short_ratio"].mean(),
                        "acct_ls": num["count_long_short_ratio"].mean(),
                        "taker_ls": num["sum_taker_long_short_vol_ratio"].mean()}
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None
    if todo:
        with ThreadPoolExecutor(24) as ex:
            rows = [r for r in ex.map(one, todo) if r]
        new = pd.DataFrame(rows).set_index("day") if rows else pd.DataFrame()
        have = pd.concat([have, new]).sort_index()
        os.makedirs(CACHE, exist_ok=True)
        have.to_csv(path)
    return have


def fear_greed() -> pd.Series:
    d = _cached_json("fear_greed.json", lambda: requests.get("https://api.alternative.me/fng/?limit=0", timeout=30).json()["data"])
    s = pd.Series({pd.Timestamp(int(x["timestamp"]), unit="s").normalize(): float(x["value"]) for x in d})
    return s.sort_index()


def stablecoin_supply() -> pd.Series:
    d = _cached_json("stablecoins.json", lambda: requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=60).json())
    s = pd.Series({pd.Timestamp(int(x["date"]), unit="s").normalize(): x["totalCirculatingUSD"].get("peggedUSD", np.nan)
                   for x in d})
    return s.sort_index()


def dvol() -> pd.Series:
    def fetch():
        out, end = [], int(time.time() * 1000)
        start = int(pd.Timestamp("2021-03-01").value // 10**6)
        while end > start:
            r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                             params={"currency": "BTC", "start_timestamp": start, "end_timestamp": end,
                                     "resolution": "1D"}, timeout=30).json()["result"]
            if not r["data"]:
                break
            out += r["data"]
            end = r["continuation"] or 0
            if not r["continuation"]:
                break
        return out
    d = _cached_json("dvol_btc.json", fetch)
    s = pd.Series({pd.Timestamp(x[0], unit="ms").normalize(): x[4] for x in d})
    return s[~s.index.duplicated()].sort_index()


def coinbase_premium(binance_close: pd.Series) -> pd.Series:
    def fetch():
        out, t = [], pd.Timestamp("2020-01-01")
        while t < pd.Timestamp.now():
            e = t + pd.Timedelta(days=299)
            r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/candles", headers=UA,
                             params={"granularity": 86400, "start": t.isoformat(), "end": e.isoformat()}, timeout=30)
            out += r.json()
            t = e
            time.sleep(0.5)
        return out
    d = _cached_json("coinbase_btcusd.json", fetch)
    cb = pd.Series({pd.Timestamp(int(x[0]), unit="s").normalize(): float(x[4]) for x in d}).sort_index()
    cb = cb[~cb.index.duplicated()]
    return (cb / binance_close.reindex(cb.index) - 1).dropna()


COMPOSITE = {"fg_chg7": +1, "stable_30d": +1, "dvol": -1, "cb_premium3": +1, "top_ls": -1, "acct_ls": -1, "taker_ls3": +1}


def composite_test(last_month: str):
    from backend.growth_study import LIVE, load_market_bulk, run_account
    from backend.strategy_lab import stats
    from backend.trend_strategy import compute_features
    from backend.universe_data import available_months, load_bars
    fg, stable, vol = fear_greed(), stablecoin_supply(), dvol()
    btc_close = load_bars("BTCUSDT", "1d", last_month)["close"]
    btc_close.index = btc_close.index.normalize()
    prem = coinbase_premium(btc_close)
    days = pd.date_range("2020-01-01", pd.Timestamp.now().normalize())
    market_wide = pd.DataFrame({"fg_chg7": fg - fg.shift(7), "stable_30d": stable / stable.shift(30) - 1,
                                "dvol": vol, "cb_premium3": prem.rolling(3).mean()}).reindex(days).ffill(limit=3)

    def z(frame):         # vs. the trailing year, known the day before
        mu = frame.rolling(365, min_periods=90).mean()
        sd = frame.rolling(365, min_periods=90).std()
        return ((frame - mu) / sd).shift(1)
    blend, thresh = {}, {}
    for sym in UNIVERSE:
        m = binance_metrics(sym)
        m.index = pd.to_datetime(m.index)
        coin = pd.DataFrame({"top_ls": m["top_ls"], "acct_ls": m["acct_ls"],
                             "taker_ls3": m["taker_ls"].rolling(3).mean()}).reindex(days)
        zs = z(pd.concat([market_wide, coin], axis=1))
        b = sum(zs[k] * sign for k, sign in COMPOSITE.items()) / zs[list(COMPOSITE)].notna().sum(axis=1).replace(0, np.nan)
        blend[sym] = b
        thresh[sym] = b.rolling(365, min_periods=90).quantile(1 / 3).shift(1)

    calls = {"n": 0, "blocked": 0}

    def can_enter(sym, ts):
        day = pd.Timestamp(ts + 4 * 3_600_000, unit="ms").normalize()
        b, t = blend[sym].get(day, np.nan), thresh[sym].get(day, np.nan)
        calls["n"] += 1
        if np.isnan(b) or np.isnan(t) or b >= t:
            return True
        calls["blocked"] += 1
        return False

    market = load_market_bulk("4h", last_month, UNIVERSE, lambda s: available_months(s, "4h"))
    feats = {s: compute_features(m["bars"], LIVE) for s, m in market.items()}
    timeline = sorted({b["timestamp"] for m in market.values() for b in m["bars"]})
    t0 = int(pd.Timestamp("2021-01-01").value // 10**6)

    def daily(fn):
        curve: list = []
        run_account(market, feats, timeline, LIVE, t0, timeline[-1] + 1, liquidation=False, can_enter=fn, curve=curve)
        eq = pd.Series(dict(curve))
        eq.index = pd.to_datetime(eq.index, unit="ms")
        return eq.resample("1D").last().dropna().pct_change().dropna()

    def line(name, r):
        a, b = stats(r[:"2024-06-30"]), stats(r["2024-07-01":])
        return (f"  {name:<26} 2021-24H1: CAGR {a['cagr']:+6.1%} Sharpe {a['sharpe']:+.2f} DD {a['maxdd']:+6.1%} | "
                f"2024H2+: CAGR {b['cagr']:+6.1%} Sharpe {b['sharpe']:+.2f} DD {b['maxdd']:+6.1%}")
    base = daily(None)
    filt = daily(can_enter)
    rate = calls["blocked"] / max(calls["n"], 1)
    print(f"Blend filter blocks {rate:.0%} of breakout signals. Live trend strategy, 1% risk:")
    print(line("unfiltered", base))
    print(line("blend filter", filt))
    import zlib
    for seed in range(5):
        print(line(f"random skips, seed {seed}",
                   daily(lambda s, ts, seed=seed: zlib.crc32(f"{s}{ts}{seed}".encode()) % 10_000 / 10_000 >= rate)))


def main():
    if "--composite" in __import__("sys").argv:
        return composite_test(time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400)))
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    trades = live_trend_trades(last_month)
    print(f"{len(trades)} trend trades on the 8 coins", flush=True)

    metrics: Dict[str, pd.DataFrame] = {}
    for s in UNIVERSE:
        metrics[s] = binance_metrics(s)
        print(f"  metrics {s}: {len(metrics[s])} days from {metrics[s].index.min():%Y-%m-%d}", flush=True)
    fg, stable, vol = fear_greed(), stablecoin_supply(), dvol()
    from backend.universe_data import load_bars
    btc_close = load_bars("BTCUSDT", "1d", last_month)["close"]
    btc_close.index = btc_close.index.normalize()
    prem = coinbase_premium(btc_close)
    from backend.universe_data import daily_panel
    closes = daily_panel(last_month)["close"]
    listed = closes.apply(lambda c: c.first_valid_index()).dropna()
    new_per_day = listed.dt.normalize().value_counts().reindex(closes.index, fill_value=0).sort_index()
    new30 = new_per_day.rolling(30, min_periods=1).sum()
    active = closes.notna().sum(axis=1)
    active.index, new30.index = active.index.normalize(), new30.index.normalize()
    print(f"  fear&greed {len(fg)} days, stablecoins {len(stable)}, DVOL {len(vol)}, Coinbase premium {len(prem)}", flush=True)

    def at(s: pd.Series, day, lag=1):
        v = s.get(day - pd.Timedelta(days=lag))
        return np.nan if v is None else float(v)

    def chg(s: pd.Series, day, days):
        a, b = at(s, day, 1), at(s, day, 1 + days)
        return a / b - 1 if a and b and not np.isnan(a) and not np.isnan(b) else np.nan

    rows = []
    for t in trades.itertuples():
        day = pd.Timestamp(t.entry_s, unit="s").normalize()
        m = metrics[t.sym]
        p3 = prem[day - pd.Timedelta(days=3):day - pd.Timedelta(days=1)]
        rows.append({
            "fear_greed": at(fg, day), "fg_chg7": at(fg, day) - at(fg, day, 8),
            "stable_30d": chg(stable, day, 30), "dvol": at(vol, day), "dvol_chg7": chg(vol, day, 7),
            "cb_premium3": p3.mean() if len(p3) else np.nan,
            "oi_chg3": chg(m["oi_value"], day, 3), "oi_chg30": chg(m["oi_value"], day, 30),
            "top_ls": at(m["top_ls"], day), "acct_ls": at(m["acct_ls"], day),
            "taker_ls3": m["taker_ls"][day - pd.Timedelta(days=3):day - pd.Timedelta(days=1)].mean(),
            "new_listings_30d": at(new30, day), "active_perps": at(active, day),
        })
    feats = pd.DataFrame(rows)
    trades = pd.concat([trades, feats], axis=1)
    rng = np.random.default_rng(0)
    r = trades["r"].to_numpy()
    oos = (trades["entry_s"] >= _ms(SPLIT) // 1000).to_numpy()
    for label, mask in (("2020-03 .. 2024-06", ~oos), ("2024-07 .. now", oos)):
        print(f"\n{label}: E[R] {r[mask].mean():+.2f}. Top vs bottom third of each measure at entry:")
        for name in feats.columns:
            print(split_test(name, trades[name].to_numpy()[mask], r[mask], rng))
    os.makedirs(CACHE, exist_ok=True)
    trades.to_csv(os.path.join(CACHE, "trend_trades_insight.csv"), index=False)


if __name__ == "__main__":
    main()
