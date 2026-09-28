"""
Coinbase premium across coins: when US buyers on Coinbase pay more for a coin than traders on Binance, is that real
demand that continues? insight_lab.py used Bitcoin's Coinbase premium as a timing filter (only significant in one
period); this is the cross-sectional version. Fixed in advance:

  premium   log(Coinbase USD daily close / Binance USDT spot daily close), both at 00:00 UTC; the USDT/USD rate is
            common to all coins, so only the ranking across coins matters
  signal    7-day mean premium
  universe  point-in-time top 30 Binance USDT perps that also trade on Coinbase (USD)
  book      7 daily slices, long 4 / short 4 on Binance perps, 0.1% per unit of turnover, real funding
  pass      Sharpe > 0.5 with funding in both 2022-23 and 2024+
  caveat    Coinbase's product list is today's (coins it delisted are missing)

    python -m backend.cb_premium_lab
"""

import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from backend import positioning_lab as pl

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "coinbase"))


def cb_closes(base: str) -> pd.Series:
    path = os.path.join(DATA, f"cb_{base}.json")
    if not os.path.exists(path):
        rows, start = [], pd.Timestamp("2021-10-01")
        while start < pd.Timestamp.now():
            end = min(start + pd.Timedelta(days=299), pd.Timestamp.now().normalize())
            for k in range(4):
                r = requests.get(f"https://api.exchange.coinbase.com/products/{base}-USD/candles",
                                 params={"granularity": 86400, "start": start.isoformat() + "Z", "end": end.isoformat() + "Z"},
                                 timeout=30)
                if r.status_code == 200:
                    rows += r.json()
                    break
                time.sleep(2 * (k + 1))
            time.sleep(0.25)
            start = end + pd.Timedelta(days=1)
        json.dump(rows, open(path, "w"))
    rows = json.load(open(path))
    s = pd.Series({pd.Timestamp(int(x[0]), unit="s"): float(x[4]) for x in rows})
    return s[~s.index.duplicated()].sort_index()


def spot_closes(sym: str) -> pd.Series:
    path = os.path.join(DATA, f"spot_{sym}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    months = pd.period_range("2021-10", pd.Timestamp.now() - pd.Timedelta(days=32), freq="M")

    def one(m):
        url = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1d/{sym}-1d-{m}.zip"
        for k in range(3):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 404:
                    return []
                z = zipfile.ZipFile(io.BytesIO(r.content))
                return [line.split(",")[:5] for line in z.read(z.namelist()[0]).decode().splitlines() if line[:1].isdigit()]
            except Exception:                              # noqa: BLE001
                time.sleep(2 * (k + 1))
        return []
    with ThreadPoolExecutor(8) as ex:
        rows = [r for rs in ex.map(one, [str(m) for m in months]) for r in rs]
    ts = [int(r[0]) // (1000 if int(r[0]) > 1e14 else 1) for r in rows]           # 2025+ files are in microseconds
    s = pd.Series([float(r[4]) for r in rows], index=pd.to_datetime(ts, unit="ms")).sort_index()
    s = s[~s.index.duplicated()]
    s.to_frame("close").to_csv(path)
    return s


def main():
    os.makedirs(DATA, exist_ok=True)
    sig, ret, member, _ = pl.signals()
    products = requests.get("https://api.exchange.coinbase.com/products", timeout=30).json()
    usd = {x["base_currency"] for x in products if x["quote_currency"] == "USD"}
    syms = [s for s in ret.columns[member.any().to_numpy()] if s.endswith("USDT") and s[:-4].replace("1000", "") in usd
            and not s.startswith("1000")]
    print(f"{len(syms)} top-30 coins also trade on Coinbase", flush=True)
    prem = {}
    for s in syms:
        cb, sp = cb_closes(s[:-4]), spot_closes(s)
        both = pd.DataFrame({"cb": cb, "bn": sp}).dropna()
        if len(both) > 60:
            prem[s] = np.log(both["cb"] / both["bn"])
    p = pd.DataFrame(prem).reindex(ret.index)
    signal = p.rolling(7, min_periods=5).mean().reindex(columns=ret.columns)
    m = member & signal.notna()
    print(f"eligible coins on a typical day: {m.sum(axis=1)['2022':].median():.0f}; "
          f"median |premium| {p.abs().stack().median() * 1e4:.0f} bp")
    held = sorted(set(ret.columns[m.any().to_numpy()]))
    fund = pl.daily_funding(held, ret.index).reindex(columns=ret.columns).fillna(0.0)

    def book(s, **kw):
        return pd.concat([pl.backtest(s, ret, m, legs=4, offset=o, **kw) for o in range(7)], axis=1).mean(axis=1)
    print(f"  {'price only':22s} {pl._fmt(book(signal))}")
    print(f"  {'with funding':22s} {pl._fmt(book(signal, funding=fund))}")
    print(f"  {'opposite sign (check)':22s} {pl._fmt(book(-signal, funding=fund))}")
    z = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
    print(f"  correlation with the smart-money signal: {z(signal.where(m)).corrwith(z(sig['smart'].where(m)), axis=1).mean():+.2f}")


if __name__ == "__main__":
    main()
