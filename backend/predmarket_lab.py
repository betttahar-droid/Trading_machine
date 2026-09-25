"""
Prediction markets: is there a behavioural edge a small account can harvest?

The favorite-longshot bias: people overpay for long shots and underpay for favourites, so contracts priced
at 90c should win more than 90% of the time. Tested on resolved Polymarket binary markets (public gamma API +
CLOB price history, no key needed):

- A market is "bought" at a fixed time before its scheduled end date (endDate - h, known in advance, so no
  look-ahead on when it actually resolved); markets already closed by then are skipped.
- Calibration: how often a side priced p actually won.
- Strategies: buy the favourite side when it is priced in a band (e.g. 80-97c), or the longshot, pay a
  1c spread, hold to resolution. Return per trade and per day held.
Split by resolution date: before 2025-07 vs 2025-07 on.

Check that prediction markets are legal where you live before trading them (Polymarket is blocked in several
countries, e.g. France).

    python -m backend.predmarket_lab
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from backend.backtest_trend import DATA_DIR

CACHE = os.path.join(DATA_DIR, "predmarket_cache")
GAMMA = "https://gamma-api.polymarket.com/markets/keyset"
CLOB = "https://clob.polymarket.com/prices-history"
SPREAD = 0.01          # paid on entry (buy at mid + 1c)
MIN_VOLUME = 50_000
HOURS = (24, 24 * 7, 24 * 30)          # buy this long before the scheduled end (12h price resolution)


def _get(url, params, tries=5):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=40)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * (attempt + 1))
    return None


def resolved_markets(max_pages: int = 600) -> pd.DataFrame:
    """Closed markets by volume, largest first, down to MIN_VOLUME (keyset pagination, 100 per page)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "markets.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 7 * 86400:
        rows = json.load(open(path))
    else:
        rows, cursor = [], None
        for page in range(max_pages):
            params = {"closed": "true", "limit": 100, "order": "volumeNum", "ascending": "false"}
            if cursor:
                params["after_cursor"] = cursor
            d = _get(GAMMA, params)
            if not d or not d.get("markets"):
                break
            keep = ("id", "question", "outcomes", "outcomePrices", "clobTokenIds", "volumeNum", "endDate",
                    "closedTime", "negRisk")
            rows += [{k: m.get(k) for k in keep} for m in d["markets"]]
            cursor = d.get("next_cursor")
            if not cursor or (d["markets"][-1].get("volumeNum") or 0) < MIN_VOLUME:
                break
            if page % 50 == 49:
                print(f"  fetched {len(rows)} markets, volume down to ${d['markets'][-1].get('volumeNum', 0):,.0f}", flush=True)
        json.dump(rows, open(path, "w"))
    out = []
    for m in rows:
        try:
            outcomes, prices, tokens = (json.loads(m[k]) for k in ("outcomes", "outcomePrices", "clobTokenIds"))
        except (KeyError, TypeError, ValueError):
            continue
        if len(outcomes) != 2 or sorted(prices) != ["0", "1"] or (m.get("volumeNum") or 0) < MIN_VOLUME:
            continue
        end, closed = pd.to_datetime(m.get("endDate"), utc=True, errors="coerce"), pd.to_datetime(m.get("closedTime"), utc=True, errors="coerce")
        if pd.isna(end) or pd.isna(closed):
            continue
        out.append({"id": m["id"], "question": m["question"], "token": tokens[0], "won": int(prices[0] == "1"),
                    "end": end, "closed": closed, "volume": m["volumeNum"], "negrisk": bool(m.get("negRisk"))})
    return pd.DataFrame(out)


def price_history(token: str) -> Optional[pd.Series]:
    path = os.path.join(CACHE, f"hist_{token[:40]}.json")
    if os.path.exists(path):
        pts = json.load(open(path))
    else:
        d = _get(CLOB, {"market": token, "interval": "max", "fidelity": 720})   # 12h points (finer returns nothing)
        pts = d.get("history", []) if d else []
        json.dump(pts, open(path, "w"))
    if not pts:
        return None
    s = pd.Series({pd.Timestamp(p["t"], unit="s", tz="UTC"): float(p["p"]) for p in pts})
    return s.sort_index()


def build(markets: pd.DataFrame) -> pd.DataFrame:
    with ThreadPoolExecutor(8) as ex:
        hists = list(ex.map(price_history, markets["token"]))
    rows = []
    for m, h in zip(markets.itertuples(), hists):
        if h is None or len(h) < 3:
            continue
        for hours in HOURS:
            t0 = m.end - pd.Timedelta(hours=hours)
            if m.closed <= t0 or h.index[0] > t0:          # already closed, or not trading yet
                continue
            p = h[:t0].iloc[-1]
            if not 0.005 < p < 0.995:
                continue
            rows.append({"id": m.id, "question": m.question, "hours": hours, "p": p, "won": m.won,
                         "hold_days": max((m.closed - t0).total_seconds() / 86400, 0.5),
                         "resolved": m.closed, "negrisk": m.negrisk, "volume": m.volume})
    return pd.DataFrame(rows)


def trade_side(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Buy whichever side is priced in [lo, hi] (YES at p, or NO at 1-p), paying the spread."""
    yes = df[(df["p"] >= lo) & (df["p"] <= hi)].assign(price=lambda d: d["p"], win=lambda d: d["won"])
    no = df[(1 - df["p"] >= lo) & (1 - df["p"] <= hi)].assign(price=lambda d: 1 - d["p"], win=lambda d: 1 - d["won"])
    t = pd.concat([yes, no])
    cost = (t["price"] + SPREAD).clip(upper=0.999)
    return t.assign(ret=t["win"] / cost - 1)


def main():
    markets = resolved_markets()
    markets = markets[markets["closed"] >= pd.Timestamp("2023-01-01", tz="UTC")]
    print(f"{len(markets)} resolved binary Polymarket markets with volume >= ${MIN_VOLUME:,} since 2023", flush=True)
    df = build(markets)
    df.to_csv(os.path.join(CACHE, "snapshots.csv"), index=False)
    split = pd.Timestamp("2025-07-01", tz="UTC")
    for hours in HOURS:
        d = df[df["hours"] == hours]
        print(f"\n=== bought {hours // 24} day(s) before the scheduled end: {d['id'].nunique()} markets ===")
        both = pd.concat([d.assign(price=d["p"], win=d["won"]), d.assign(price=1 - d["p"], win=1 - d["won"])])
        both["bucket"] = pd.cut(both["price"], [0, .05, .1, .2, .35, .5, .65, .8, .9, .95, 1])
        cal = both.groupby("bucket", observed=True).agg(n=("win", "size"), price=("price", "mean"), won=("win", "mean"))
        cal["edge"] = cal["won"] - cal["price"]
        print("calibration (both sides of every market): priced -> actually won")
        print(cal.to_string(float_format=lambda v: f"{v:.3f}"))
        for lo, hi, label in ((0.80, 0.97, "favourites 80-97c"), (0.90, 0.98, "favourites 90-98c"),
                              (0.02, 0.20, "longshots 2-20c")):
            for name, part in (("before 2025-07", d[d["resolved"] < split]), ("2025-07 on", d[d["resolved"] >= split])):
                t = trade_side(part, lo, hi)
                if t.empty:
                    continue
                per_day = t["ret"].sum() / t["hold_days"].sum()
                print(f"  {label:<18} {name:<15} n={len(t):5d} win {t['win'].mean():5.1%} "
                      f"mean return {t['ret'].mean():+6.2%} per trade, median hold {t['hold_days'].median():5.1f} d, "
                      f"~{per_day:+.3%} per day held, worst stretch: {(t['ret'] < 0).sum()} losses")


if __name__ == "__main__":
    main()
