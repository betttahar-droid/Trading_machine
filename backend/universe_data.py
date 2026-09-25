"""
Point-in-time coin universe from the Binance USD-M futures archive (data.binance.vision).

Every USDT perpetual that ever traded is included, delisted ones too (LUNA, FTT, SRM, ...), so a
backtest that picks "the top N coins by volume" at each date only uses what was known at that date
instead of today's survivors.

    python -m backend.universe_data            # download 1d klines for all USDT perps (cached)
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

from backend.backtest_trend import DATA_DIR
from backend.growth_study import BULK_CACHE, _bulk_csv, _rows

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = "data/futures/um/monthly"
# Stablecoins and index products, not tradable trends
EXCLUDE = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "USDPUSDT", "USDEUSDT", "EURUSDT", "BTCDOMUSDT",
           "DEFIUSDT", "BLUEBIRDUSDT", "FOOTBALLUSDT", "ALLUSDT"}


def _s3_list(prefix: str, pattern: str) -> List[str]:
    out, marker = [], ""
    while True:
        r = requests.get(S3, params={"delimiter": "/", "prefix": prefix, "marker": marker}, timeout=30)
        r.raise_for_status()
        found = re.findall(pattern, r.text)
        out += found
        if "<IsTruncated>true</IsTruncated>" not in r.text or not found:
            return out
        marker = re.findall(r"<(?:Key|Prefix)>([^<]+)</(?:Key|Prefix)>", r.text)[-1]


def usdt_perps() -> List[str]:
    path = os.path.join(BULK_CACHE, "um_symbols.txt")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 7 * 86400:
        return open(path).read().split()
    syms = _s3_list(f"{PREFIX}/klines/", rf"<Prefix>{PREFIX}/klines/([^/]+)/</Prefix>")
    syms = [s for s in syms if s.endswith("USDT") and s not in EXCLUDE]
    os.makedirs(BULK_CACHE, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(syms))
    return syms


def available_months(symbol: str, interval: str) -> List[str]:
    path = os.path.join(BULK_CACHE, "months", f"{symbol}-{interval}.txt")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 7 * 86400:
        return open(path).read().split()
    keys = _s3_list(f"{PREFIX}/klines/{symbol}/{interval}/", rf"{symbol}-{interval}-(\d{{4}}-\d{{2}})\.zip<")
    months = sorted(set(keys))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(months))
    return months


def load_bars(symbol: str, interval: str, last_month: str) -> pd.DataFrame:
    """OHLC + quote volume for one symbol, from its listing month to last_month."""
    months = [m for m in available_months(symbol, interval) if m <= last_month]
    rows = []
    for m in months:
        for r in _rows(_bulk_csv("klines", symbol, m, interval)):
            rows.append((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[7])))
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "qvol"]).drop_duplicates("t")
    df.index = pd.to_datetime(df.pop("t"), unit="ms")
    return df.sort_index()


def load_funding(symbol: str, months: List[str]) -> pd.Series:
    rows = [(int(r[0]), float(r[-1])) for m in months for r in _rows(_bulk_csv("funding", symbol, m, "4h"))]
    s = pd.Series(dict(rows), dtype=float)
    s.index = pd.to_datetime(s.index, unit="ms")
    return s.sort_index()


def daily_panel(last_month: str, threads: int = 24) -> Dict[str, pd.DataFrame]:
    """close / qvol / open / high / low DataFrames (dates x symbols) for every USDT perp."""
    syms = usdt_perps()
    with ThreadPoolExecutor(threads) as ex:
        frames = dict(zip(syms, ex.map(lambda s: load_bars(s, "1d", last_month), syms)))
    frames = {s: f for s, f in frames.items() if len(f) > 0}
    return {col: pd.DataFrame({s: f[col] for s, f in frames.items()}).sort_index()
            for col in ("open", "high", "low", "close", "qvol")}


def top_by_volume(qvol: pd.DataFrame, close: pd.DataFrame, n: int, lookback_days: int = 30,
                  min_history_days: int = 60) -> pd.DataFrame:
    """Boolean (dates x symbols): in the top n by trailing quote volume, using data up to the prior day."""
    vol = qvol.rolling(lookback_days, min_periods=lookback_days).mean().shift(1)
    history = close.notna().cumsum().shift(1) >= min_history_days
    vol = vol.where(history & close.notna())
    rank = vol.rank(axis=1, ascending=False, method="first")
    return rank <= n


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    t0 = time.time()
    syms = usdt_perps()
    print(f"{len(syms)} USDT perps in the archive (incl. delisted)", flush=True)
    panel = daily_panel(last_month)
    print(f"daily panel {panel['close'].shape} in {time.time() - t0:.0f}s", flush=True)
    top = top_by_volume(panel["qvol"], panel["close"], 30)
    ever = top.any()
    print(f"{int(ever.sum())} symbols were ever in the top 30 by volume; e.g. {list(ever[ever].index[:40])}")


if __name__ == "__main__":
    main()
