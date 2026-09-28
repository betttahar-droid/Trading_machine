"""
Korean retail frenzies: Upbit (South Korea's largest exchange, mostly retail, KRW only) regularly sees coin-specific
manias. If a coin's trading suddenly shifts to Upbit, the crowd is piling in; the smart-money finding says to lean
against the crowd. Fixed in advance:

  share     Upbit KRW daily volume / Binance USDT perp daily quote volume (the KRW/USD rate is the same for every coin
            on a day, so only the ranking across coins matters)
  signal    minus [log(mean share over the last 7 days) - log(mean share over the 60 days before)]: short the coins
            whose Korean share surged, long those Korea is losing interest in
  universe  point-in-time top 30 Binance USDT perps that trade on Upbit in KRW (today's Upbit list)
  book      7 daily slices, long 5 / short 5, 0.1% per unit of turnover, real funding
  pass      Sharpe > 0.5 with funding in both 2022-23 and 2024+
  checks    the level of the share (not its change), and the opposite sign

    python -m backend.upbit_lab
"""

import json
import os
import time

import numpy as np
import pandas as pd
import requests

from backend import positioning_lab as pl
from backend.universe_data import daily_panel

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "upbit"))
API = "https://api.upbit.com/v1"


def candles(market: str) -> pd.Series:
    path = os.path.join(DATA, f"{market}.json")
    if not os.path.exists(path):
        rows, to = [], None
        while True:
            params = {"market": market, "count": 200}
            if to:
                params["to"] = to
            for k in range(5):
                r = requests.get(f"{API}/candles/days", params=params, timeout=30)
                if r.status_code == 200:
                    page = r.json()
                    break
                time.sleep(1 + 2 * k)
            else:
                page = []
            time.sleep(0.15)
            if not page:
                break
            rows += page
            oldest = min(x["candle_date_time_utc"] for x in page)
            if oldest < "2021-09-01" or len(page) < 200:
                break
            to = oldest.replace("T", " ")
        json.dump(rows, open(path, "w"))
    rows = json.load(open(path))
    s = pd.Series({pd.Timestamp(x["candle_date_time_utc"]): float(x["candle_acc_trade_price"]) for x in rows})
    return s[~s.index.duplicated()].sort_index()


def main():
    os.makedirs(DATA, exist_ok=True)
    sig, ret, member, _ = pl.signals()
    krw = {m["market"][4:] for m in requests.get(f"{API}/market/all", timeout=30).json() if m["market"].startswith("KRW-")}
    syms = [s for s in ret.columns[member.any().to_numpy()] if s[:-4].replace("1000", "") in krw]
    print(f"{len(syms)} top-30 coins trade on Upbit (KRW)", flush=True)
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    qvol = daily_panel(last_month)["qvol"]
    share = {}
    for s in syms:
        up = candles("KRW-" + s[:-4].replace("1000", ""))
        if len(up) > 90:
            share[s] = (up / qvol[s]).replace([np.inf, -np.inf], np.nan)
    sh = pd.DataFrame(share).reindex(ret.index)
    recent = sh.rolling(7, min_periods=5).mean()
    before = sh.shift(7).rolling(60, min_periods=40).mean()
    surge = -(np.log(recent) - np.log(before))
    level = -np.log(recent)
    signal = surge.reindex(columns=ret.columns)
    m = member & signal.notna()
    print(f"eligible coins on a typical day: {m.sum(axis=1)['2022':].median():.0f}")
    held = sorted(set(ret.columns[m.any().to_numpy()]))
    fund = pl.daily_funding(held, ret.index).reindex(columns=ret.columns).fillna(0.0)

    def book(s, **kw):
        return pd.concat([pl.backtest(s.reindex(columns=ret.columns), ret, m, legs=5, offset=o, **kw)
                          for o in range(7)], axis=1).mean(axis=1)
    print(f"  {'surge, price only':32s} {pl._fmt(book(signal))}")
    print(f"  {'surge, with funding':32s} {pl._fmt(book(signal, funding=fund))}")
    print(f"  {'opposite sign (check)':32s} {pl._fmt(book(-signal, funding=fund))}")
    print(f"  {'share level, with funding':32s} {pl._fmt(book(level, funding=fund))}")
    z = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
    print(f"  correlation with the smart-money signal: {z(signal.where(m)).corrwith(z(sig['smart'].where(m)), axis=1).mean():+.2f}")


if __name__ == "__main__":
    main()
