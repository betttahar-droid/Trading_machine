"""
Crypto sector momentum: rotate between groups of coins (layer-1s, layer-2s, memes, AI, DeFi, gaming, real-world assets,
privacy) instead of single coins. Single-coin momentum failed out of sample (strategy_lab.py); sector averages are
less noisy. Fixed in advance:

  sectors  CoinGecko category members (today's membership: a known look-ahead), one sector per coin in the priority
           order below, traded through Binance USDT perps in the point-in-time top 100 by volume (delisted included)
  rule     every 7 days rank sectors by their 28-day return; long the top 2 and short the bottom 2 (equal weight);
           long-only variant: top 2 vs holding all sectors; 0.1% cost per unit of turnover
  test     2021-01 .. 2023-12 vs 2024-01 .. now

    python -m backend.sector_lab
"""

import json
import os
import time

import numpy as np
import pandas as pd
import requests

from backend.universe_data import daily_panel, top_by_volume

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sectors"))
PRIORITY = [("meme-token", "Meme"), ("artificial-intelligence", "AI"), ("gaming", "Gaming"),
            ("real-world-assets-rwa", "RWA"), ("privacy-coins", "Privacy"), ("layer-2", "L2"),
            ("decentralized-finance-defi", "DeFi"), ("layer-1", "L1")]
SPLIT, COST = "2024-01-01", 0.001


def members() -> dict:
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "members.json")
    if os.path.exists(path):
        return json.load(open(path))
    out = {}
    for cat, name in PRIORITY:
        syms = set()
        for page in (1, 2):
            for k in range(6):
                r = requests.get("https://api.coingecko.com/api/v3/coins/markets",
                                 params={"vs_currency": "usd", "category": cat, "per_page": 250, "page": page}, timeout=30)
                if r.status_code == 200:
                    syms |= {c["symbol"].upper() for c in r.json()}
                    break
                time.sleep(20 * (k + 1))
            time.sleep(15)
        out[name] = sorted(syms)
        print(f"  {name}: {len(syms)} coins", flush=True)
    json.dump(out, open(path, "w"))
    return out


def main():
    mem = members()
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    panel = daily_panel(last_month)
    close = panel["close"]
    top = top_by_volume(panel["qvol"], close, 100)
    base = {s: s[:-4].replace("1000000", "").replace("1000", "") for s in close.columns if s.endswith("USDT")}
    sector_of = {}
    for _, name in PRIORITY:
        for s, b in base.items():
            if s not in sector_of and b in mem[name]:
                sector_of[s] = name
    ret = close.pct_change(fill_method=None).where(top.shift(1).fillna(False))
    sec = pd.DataFrame({name: ret[[s for s, n in sector_of.items() if n == name]].mean(axis=1)
                        for _, name in PRIORITY}).loc["2020-10-01":]
    counts = pd.DataFrame({name: ret[[s for s, n in sector_of.items() if n == name]].notna().sum(axis=1)
                           for _, name in PRIORITY}).loc["2020-10-01":]
    print("average coins per sector in the top 100:", counts.loc["2021":].mean().round(1).to_dict())
    sec = sec.where(counts >= 2)
    mom = (1 + sec.fillna(0)).rolling(28).apply(np.prod, raw=True) - 1
    rows = []
    days = sec.index[(sec.index >= "2021-01-01")]
    w_prev = pd.Series(0.0, index=sec.columns)
    for k, d in enumerate(days):
        if k % 7 == 0:
            m = mom.loc[d].where(counts.loc[d] >= 2).dropna()
            w = pd.Series(0.0, index=sec.columns)
            wl = pd.Series(0.0, index=sec.columns)
            if len(m) >= 4:
                ranked = m.sort_values()
                w[ranked.index[-2:]] = 0.5
                w[ranked.index[:2]] = -0.5
                wl[ranked.index[-2:]] = 0.5
            turn = (w - w_prev).abs().sum()
            w_prev, w_long = w, wl
        else:
            turn = 0.0
        nxt = sec.shift(-1).loc[d].fillna(0)
        rows.append({"d": d, "ls": (w_prev * nxt).sum() - turn * COST, "long_top2": (w_long * nxt).sum() - turn * COST / 2,
                     "all_sectors": nxt[counts.loc[d] >= 2].mean() if (counts.loc[d] >= 2).any() else 0.0})
    r = pd.DataFrame(rows).set_index("d")
    for col, lab in (("ls", "long top 2 / short bottom 2"), ("long_top2", "long top 2 sectors"), ("all_sectors", "hold all sectors")):
        out = []
        for per, x in (("2021-23", r[col][:SPLIT]), ("2024+", r[col][SPLIT:])):
            eq = (1 + x).cumprod()
            out.append(f"{per}: {eq.iloc[-1] ** (365 / len(x)) - 1:+.1%}/yr Sharpe {x.mean() / x.std() * np.sqrt(365):+.2f} "
                       f"DD {(eq / eq.cummax() - 1).min():+.0%}")
        print(f"  {lab:28s} " + " | ".join(out))


if __name__ == "__main__":
    main()
