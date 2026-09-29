"""
Year-by-year history of the live plan since 2015, at risk levels 1-3 (answer to "how much would it have won each year").

  crypto   live trend rules (4h Donchian 120/60, 4x ATR chandelier, long-only, 8 coins, risk 0.68% x level) via
           backtest_trend.simulate. Futures data from 2020-01; before that Binance SPOT 4h candles (Binance opened
           2017-07, so crypto starts 2017-08; no funding on spot). Coins join when they were listed.
  tradfi   tradfi_book rules on the ETFs (12-month momentum, 60-day vol, target 14.6% x level, caps 1.5x / 3x),
           rebalanced at month end, 0.05% cost, perp funding at 2025-26 rates (PAXG gold ~4%/yr).
  plan     both books on one account (each sizes off the whole equity), so daily returns add.

Caveats: the 8 coins are today's survivors (SOL, AVAX, SUI picked in hindsight); the TradFi perps only exist since
2026, so earlier years assume they had existed at today's funding.

    python -m backend.history_lab
"""

import io
import os
import time
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import requests

from backend.backtest_trend import UNIVERSE, _ms, simulate
from backend.cross_asset_lab import FUNDING_2026, etf_prices
from backend.growth_study import BULK_CACHE, LIVE, load_market_bulk
from backend import tradfi_book as tb
from backend.universe_data import available_months

SPOT_URL = "https://data.binance.vision/data/spot/monthly/klines"
FUNDING = dict(FUNDING_2026, GLD=0.04)
COST = 0.0005


def _spot_bars(sym: str, last: str = "2019-12") -> list:
    bars, y, m = [], 2017, 8
    while f"{y:04d}-{m:02d}" <= last:
        name = f"{sym}-4h-{y:04d}-{m:02d}"
        path = os.path.join(BULK_CACHE, "spot_" + name + ".csv")
        if not os.path.exists(path):
            r = requests.get(f"{SPOT_URL}/{sym}/4h/{name}.zip", timeout=30)
            text = "" if r.status_code == 404 else zipfile.ZipFile(io.BytesIO(r.content)).read(name + ".csv").decode()
            os.makedirs(BULK_CACHE, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = line.split(",")
                if r[0][:1].isdigit():
                    bars.append({"timestamp": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                                 "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])})
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return bars


def crypto_market() -> dict:
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    fut = load_market_bulk("4h", last_month, UNIVERSE, lambda s: available_months(s, "4h"))
    market = {}
    for sym in UNIVERSE:
        f = fut.get(sym, {"bars": [], "funding": np.zeros(0)})
        first = f["bars"][0]["timestamp"] if f["bars"] else 10 ** 15
        spot = [b for b in _spot_bars(sym) if b["timestamp"] < first]
        bars = spot + f["bars"]
        if len(bars) < 150:
            continue
        fund = np.concatenate([np.zeros(len(spot)), f["funding"]])
        market[sym] = {"bars": bars, "funding": fund, "idx": {b["timestamp"]: i for i, b in enumerate(bars)}}
    return market


def crypto_returns(market: dict, level: float) -> pd.Series:
    p = replace(LIVE, risk_pct=tb.CRYPTO_RISK_PER_LEVEL * level)
    end = max(b["timestamp"] for m in market.values() for b in m["bars"]) + 1
    curve: list = []
    simulate(market, p, _ms("2017-08-01"), end, curve_out=curve)
    eq = pd.Series(dict(curve))
    eq.index = pd.to_datetime(eq.index, unit="ms")
    return eq.resample("1D").last().dropna().pct_change().dropna()


def tradfi_returns(px: pd.DataFrame, level: float) -> pd.Series:
    px = px[sorted(set(tb.ASSETS.values()))]
    ret = px.pct_change(fill_method=None)
    month_end = px.index.to_series().groupby(px.index.to_period("M")).transform("max") == px.index.to_series()
    rows = {}
    for d in px.index[month_end.to_numpy()]:
        h = px.loc[:d].dropna()
        if len(h) < tb.MOM_DAYS + 1:
            continue
        sig = tb.signals(h)
        w = tb.target_weights(sig, tb.TRADFI_VOL_PER_LEVEL * level)
        rows[d] = {tb.ASSETS[k]: v for k, v in w.items()}
    w = pd.DataFrame(rows).T.reindex(px.index).ffill().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    cost = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1) * COST
    cost += (held * pd.Series(FUNDING).reindex(held.columns).fillna(0.0) / 252).sum(axis=1)
    return (held * ret.fillna(0.0)).sum(axis=1) - cost


def main():
    market = crypto_market()
    print("crypto data from: " + ", ".join(f"{s.replace('USDT', '')} {pd.to_datetime(m['bars'][0]['timestamp'], unit='ms'):%Y-%m}"
                                          for s, m in market.items()))
    px = etf_prices("2013-06-01")
    for level in (1, 2, 3):
        c = crypto_returns(market, level)
        t = tradfi_returns(px, level)
        days = pd.date_range("2015-01-01", max(c.index.max(), t.index.max()), freq="D")
        c, t = c.reindex(days).fillna(0.0), t.reindex(days).fillna(0.0)
        plan = c + t
        print(f"\nLEVEL {level}   year | crypto | tradfi | PLAN | plan max drawdown in year | EUR 500 held since 2015")
        eq = 500.0
        for y in sorted(set(days.year)):
            s = plan[str(y)]
            e = (1 + s).cumprod()
            eq *= e.iloc[-1]
            print(f"          {y}{'*' if y == days.year.max() else ' '} | {(1 + c[str(y)]).prod() - 1:+6.0%} | "
                  f"{(1 + t[str(y)]).prod() - 1:+6.0%} | {e.iloc[-1] - 1:+6.0%} | {(e / e.cummax().clip(lower=1) - 1).min():+6.0%} | {eq:,.0f}")
        e = (1 + plan).cumprod()
        print(f"          whole period: {e.iloc[-1] ** (365 / len(plan)) - 1:+.0%}/yr, max drawdown {(e / e.cummax() - 1).min():+.0%}")


if __name__ == "__main__":
    main()
