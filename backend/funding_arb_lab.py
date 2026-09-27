"""
Cross-exchange funding arbitrage: long the perp where funding is cheaper, short it where funding is dearer (Binance vs
OKX), no price exposure, earn the funding difference. OKX's public API keeps only ~3 months of funding history, so
this is a short-window check (2026-07 .. 2026-08, the months both sources cover in full).

Rule (fixed in advance): per coin and day, hold the arbitrage in the direction of the past 7 days' average daily
funding difference if it exceeds 0.02%/day (~7%/yr), else flat; 0.2% per switch (4 taker trades) ; daily P&L =
position x (OKX funding - Binance funding).

    python -m backend.funding_arb_lab
"""

import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests


def okx_funding(coin: str) -> pd.Series:
    out, after = [], ""
    for _ in range(6):
        r = requests.get("https://www.okx.com/api/v5/public/funding-rate-history",
                         params={"instId": f"{coin}-USDT-SWAP", "limit": 100, **({"after": after} if after else {})},
                         timeout=20).json()
        data = r.get("data") or []
        if not data:
            break
        out += data
        after = data[-1]["fundingTime"]
        time.sleep(0.25)
    if not out:
        return pd.Series(dtype=float)
    s = pd.Series({pd.to_datetime(int(d["fundingTime"]), unit="ms"): float(d["realizedRate"] or d["fundingRate"]) for d in out})
    return s.sort_index()


def binance_funding(coin: str, months=("2026-07", "2026-08")) -> pd.Series:
    rows = []
    for m in months:
        url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{coin}USDT/{coin}USDT-fundingRate-{m}.zip"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        rows.append(pd.read_csv(z.open(z.namelist()[0])))
    if not rows:
        return pd.Series(dtype=float)
    f = pd.concat(rows)
    tcol = [c for c in f.columns if "time" in c.lower()][0]
    rcol = [c for c in f.columns if "rate" in c.lower() and "funding" in c.lower()][0]
    return pd.Series(f[rcol].to_numpy(), index=pd.to_datetime(f[tcol], unit="ms")).sort_index()


def main():
    tickers = requests.get("https://www.okx.com/api/v5/market/tickers", params={"instType": "SWAP"}, timeout=20).json()["data"]
    usdt = sorted([t for t in tickers if t["instId"].endswith("-USDT-SWAP")], key=lambda t: -float(t["volCcy24h"]) * float(t["last"]))
    coins = [t["instId"].split("-")[0] for t in usdt[:70]]
    rows = []
    for c in coins:
        o, b = okx_funding(c), binance_funding(c)
        if len(o) < 50 or len(b) < 50:
            continue
        od = o.resample("1D").sum()
        bd = b.resample("1D").sum()
        d = (od - bd).dropna()["2026-07-01":"2026-08-31"]
        if len(d) < 50:
            continue
        sig = d.rolling(7).mean().shift(1)
        pos = np.sign(sig).where(sig.abs() > 0.0002, 0.0).fillna(0.0)
        pnl = pos * d - pos.diff().abs().fillna(pos.abs()) * 0.002 / 2
        rows.append({"coin": c, "mean_diff_yr": d.mean() * 365, "abs_diff_yr": d.abs().mean() * 365,
                     "strategy_yr": pnl.mean() * 365, "days_in": (pos != 0).mean(), "switches": int((pos.diff().abs() > 0).sum())})
        time.sleep(0.2)
    r = pd.DataFrame(rows).sort_values("strategy_yr", ascending=False)
    print(f"{len(r)} coins on both exchanges, 2026-07 .. 2026-08 (daily funding, OKX minus Binance)")
    print(f"  average |difference|: {r.abs_diff_yr.mean():.1%}/yr; strategy after switching costs: mean {r.strategy_yr.mean():+.1%}/yr, "
          f"median {r.strategy_yr.median():+.1%}/yr, positive for {(r.strategy_yr > 0).mean():.0%} of coins")
    print(r.head(8).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("  ...")
    print(r.tail(4).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
