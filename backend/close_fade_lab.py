"""
Fading the US-close move. flows_lab.py found (as a side result, on BTC) that after 16:00 New York time Bitcoin gives
back part of the day's move: 16:00 is the CME Bitcoin futures settlement, the price window of the US spot ETFs
(15:00-16:00 ET benchmark) and the leveraged ETFs' rebalance, so large flows push the price there and it relaxes
afterwards. BTC was used to find it, so the independent test is ETH and SOL. Fixed before looking at them:

  signal    move = price at 16:00 ET / price at 16:00 ET the previous weekday - 1 (weekdays only)
  trade     if |move| > 1%: at 16:00 ET take the opposite side, exit at 17:00 ET (16:30 and 18:00 also shown)
  pass      ETH and SOL both positive with t >= 2 in 2024+ and positive in 2022-23; placebo hour 04:00 ET not
  costs     taker: 0.05% per side + 1 bp (BTC/ETH) or 2 bp (SOL) slippage; maker: 0.02% per side, a limit at the
            16:00 price fills only if the next 5 minutes trade through it (else no trade), exit limit likewise
            (else a taker exit at 17:05)

    python -m backend.close_fade_lab
"""

import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "minute"))
SLIP = {"BTCUSDT": 1e-4, "ETHUSDT": 1e-4, "SOLUSDT": 2e-4}
TAKER, MAKER = 0.0005, 0.0002


def minutes(sym: str) -> pd.DataFrame:
    path = os.path.join(DATA, f"{sym}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    os.makedirs(DATA, exist_ok=True)
    months = [str(m) for m in pd.period_range("2022-01", pd.Timestamp.now() - pd.Timedelta(days=32), freq="M")]

    def one(m):
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/1m/{sym}-1m-{m}.zip"
        for k in range(4):
            try:
                r = requests.get(url, timeout=60)
                if r.status_code != 200:
                    return None
                z = zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 1, 2, 3, 4])
                df = df[pd.to_numeric(df[0], errors="coerce").notna()].astype(float)
                return df
            except Exception:                              # noqa: BLE001
                continue
        return None
    with ThreadPoolExecutor(8) as ex:
        frames = [f for f in ex.map(one, months) if f is not None]
    df = pd.concat(frames)
    df.columns = ["t", "open", "high", "low", "close"]
    df["t"] = pd.to_datetime(df["t"].astype("int64") // 1000 * 1000 if df["t"].max() > 1e14 else df["t"].astype("int64"),
                             unit="ms" if df["t"].max() < 1e14 else "us")
    df = df.drop_duplicates("t").set_index("t").sort_index()
    df.to_pickle(path)
    return df


def trades(m: pd.DataFrame, sym: str, hh: int = 16, exits=((16, 30), (17, 0), (18, 0)), thr: float = 0.01) -> pd.DataFrame:
    et = m.tz_localize("UTC").tz_convert("America/New_York")
    at = lambda h, mi: et[(et.index.hour == h) & (et.index.minute == mi)]
    entry = at(hh, 0)
    entry = entry[entry.index.dayofweek < 5]
    rows = []
    prev = None
    pos = {t: i for i, t in enumerate(et.index)}
    o, hi, lo = et["open"].to_numpy(), et["high"].to_numpy(), et["low"].to_numpy()
    for t, row in entry.iterrows():
        p = row["open"]
        if prev is not None and (t - prev[0]).days <= 4:
            move = p / prev[1] - 1
            if abs(move) > thr and t in pos:
                i, side = pos[t], -np.sign(move)
                rec = {"t": t, "move": move}
                for (eh, em) in exits:
                    j = i + (eh - hh) * 60 + em
                    if j + 6 >= len(o):
                        continue
                    gross = side * (o[j] / p - 1)
                    rec[f"taker_{eh}{em:02d}"] = gross - 2 * (TAKER + SLIP[sym])
                    # maker: entry limit at p fills if the next 5 minutes trade through it; exit limit at o[j] likewise
                    fill_in = (hi[i + 1:i + 6].max() > p) if side < 0 else (lo[i + 1:i + 6].min() < p)
                    if not fill_in:
                        rec[f"maker_{eh}{em:02d}"] = np.nan
                        continue
                    fill_out = (lo[j + 1:j + 6].min() < o[j]) if side < 0 else (hi[j + 1:j + 6].max() > o[j])
                    exit_px = o[j] if fill_out else o[j + 5]
                    out_cost = MAKER if fill_out else TAKER + SLIP[sym]
                    rec[f"maker_{eh}{em:02d}"] = side * (exit_px / p - 1) - MAKER - out_cost
                rows.append(rec)
        prev = (t, p)
    return pd.DataFrame(rows).set_index("t")


def _t(x: pd.Series) -> str:
    x = x.dropna()
    return f"{1e4 * x.mean():+5.1f} bp (t {x.mean() / x.std() * np.sqrt(len(x)):+.1f}, n {len(x)})" if len(x) > 5 else "n/a"


def main():
    for sym in ("ETHUSDT", "SOLUSDT", "BTCUSDT"):
        m = minutes(sym)
        tr = trades(m, sym)
        pl = trades(m, sym, hh=4, exits=((5, 0),))
        print(f"\n{sym} ({m.index.min():%Y-%m} .. {m.index.max():%Y-%m}), fade |move| > 1% at 16:00 ET, after costs:")
        for lab, lo, hi_ in (("2022-23", "2022", "2023"), ("2024+  ", "2024", "2100")):
            g = tr.loc[lo:hi_]
            print(f"  {lab} taker ->16:30 {_t(g['taker_1630'])} | ->17:00 {_t(g['taker_1700'])} | ->18:00 {_t(g['taker_1800'])}")
            print(f"  {lab} maker ->16:30 {_t(g['maker_1630'])} | ->17:00 {_t(g['maker_1700'])} | ->18:00 {_t(g['maker_1800'])} "
                  f"| entry filled {g['maker_1700'].notna().mean():.0%}")
            gross = pl.loc[lo:hi_, "taker_500"] + 2 * (TAKER + SLIP[sym])
            print(f"  {lab} placebo 04:00 -> 05:00 ET, before costs: {_t(gross)}")


if __name__ == "__main__":
    main()
