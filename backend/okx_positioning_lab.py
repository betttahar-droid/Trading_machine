"""
Independent check of the smart-money finding (positioning_lab.py) with a different set of traders: OKX publishes its
own top-trader position ratio and all-account ratio per perpetual swap (rubik endpoints, daily, from 2024-02).
If big OKX traders leaning against the OKX crowd picks the same winners, the Binance result is less likely to be luck.

Fixed in advance:
  universe  the same point-in-time top 30 Binance USDT perps (traded on Binance, Binance funding), coins that also
            have an OKX USDT swap
  signal    3-day mean of log(OKX top-trader position ratio) - log(OKX all-account ratio); OKX's daily points are
            stamped 16:00 UTC and used only from the next Binance daily close (about a day late, to be safe)
  book      7 daily slices, long 6 / short 6, 0.1% per unit of turnover, funding (as positioning_lab)
  period    2024-02-10 .. now (all OKX has); compared with the Binance signal over the same days
  pass      Sharpe > 0.5 with funding and > 0 on price alone

    python -m backend.okx_positioning_lab collect
    python -m backend.okx_positioning_lab
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

from backend import positioning_lab as pl

DATA = os.path.join(pl.CACHE, "okx")
BASE = "https://www.okx.com/api/v5/rubik/stat/contracts/"
ENDPOINTS = {"top": "long-short-position-ratio-contract-top-trader", "acct": "long-short-account-ratio-contract"}
START_MS = 1_706_745_600_000                            # 2024-02-01


def inst(sym: str) -> str:
    base = sym[:-4]
    for pre in ("1000000", "1000"):
        if base.startswith(pre) and len(base) > len(pre):
            base = base[len(pre):]
    return f"{base}-USDT-SWAP"


def _history(inst_id: str, kind: str) -> list:
    rows, end = [], None
    while True:
        params = {"instId": inst_id, "period": "1D", "limit": 100}
        if end:
            params["end"] = end
        for k in range(5):
            try:
                r = requests.get(BASE + ENDPOINTS[kind], params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(3 * (k + 1))
                    continue
                d = r.json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(3 * (k + 1))
        else:
            return rows
        time.sleep(0.45)                                  # rubik limit: 5 requests / 2 s
        page = d.get("data") or []
        if d.get("code") != "0" or not page:
            return rows
        rows += page
        oldest = min(int(x[0]) for x in page)
        if oldest <= START_MS or len(page) < 100:
            return rows
        end = oldest


def collect():
    os.makedirs(DATA, exist_ok=True)
    close, top = pl.universe()
    syms = sorted(top.columns[top.loc["2024-01-15":].any()])
    print(f"{len(syms)} Binance symbols in the top 30 since 2024-01", flush=True)
    if os.environ.get("REVERSE"):                          # a second worker can take the list from the end
        syms = syms[::-1]
    for k, s in enumerate(syms):
        path = os.path.join(DATA, f"{s}.json")
        if os.path.exists(path):
            continue
        out = {kind: _history(inst(s), kind) for kind in ENDPOINTS}
        json.dump(out, open(path, "w"))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(syms)}", flush=True)


def okx_signal(idx) -> pd.DataFrame:
    cols = {}
    for f in os.listdir(DATA):
        d = json.load(open(os.path.join(DATA, f)))
        if not d.get("top") or not d.get("acct"):
            continue
        s = {kind: pd.Series({int(t): float(v) for t, v in d[kind]}) for kind in ENDPOINTS}
        both = pd.DataFrame(s).dropna()
        both = both[(both > 0).all(axis=1)]
        gap = np.log(both["top"]) - np.log(both["acct"])
        # stamped 16:00 UTC; usable from the next day's 00:00 close + 1 day (conservative)
        gap.index = pd.to_datetime(gap.index, unit="ms").normalize() + pd.Timedelta(days=1)
        cols[f[:-5]] = gap[~gap.index.duplicated()]
    g = pd.DataFrame(cols).sort_index()
    g = g.reindex(pd.date_range(g.index.min(), idx[-1]))
    return g.rolling(3, min_periods=2).mean().reindex(idx)


def main():
    sig, ret, member, _ = pl.signals()
    idx = ret.loc["2024-02-10":].index
    ok = okx_signal(idx).reindex(columns=ret.columns)
    ret, member, bn = ret.loc[idx], member.loc[idx], sig["smart"].loc[idx]
    cover = ok.where(member).notna().sum().sum() / member.sum().sum()
    print(f"OKX coverage of the Binance top 30: {cover:.0%} of coin-days")
    z = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
    both = member & ok.notna() & bn.notna()
    print(f"cross-sectional correlation of the OKX and Binance signals: "
          f"{z(ok.where(both)).corrwith(z(bn.where(both)), axis=1).mean():+.2f}")
    held = sorted(set(ret.columns[member.any().to_numpy()]))
    fund = pl.daily_funding(held, idx).reindex(columns=ret.columns).fillna(0.0)

    def book(s, **kw):
        return pd.concat([pl.backtest(s, ret, member, offset=o, **kw) for o in range(7)], axis=1).mean(axis=1)

    def line(r):
        eq = (1 + r).cumprod()
        halves = [r[:r.index[len(r) // 2]], r[r.index[len(r) // 2]:]]
        return (f"{eq.iloc[-1] ** (365 / len(r)) - 1:+6.1%}/yr Sharpe {r.mean() / r.std() * np.sqrt(365):+.2f} "
                f"DD {(eq / eq.cummax() - 1).min():+.0%} | halves Sharpe "
                + " / ".join(f"{h.mean() / h.std() * np.sqrt(365):+.2f}" for h in halves))
    for lab, s in (("OKX traders", ok), ("Binance traders (same days)", bn), ("average of both", (z(ok) + z(bn)) / 2)):
        print(f"  {lab:28s} price only   {line(book(s))}")
        print(f"  {'':28s} with funding {line(book(s, funding=fund))}")


if __name__ == "__main__":
    collect() if len(sys.argv) > 1 and sys.argv[1] == "collect" else main()
