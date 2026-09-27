"""
Token unlocks from free data: CoinGecko's market-cap history divided by price gives each coin's circulating supply
per day (keyless API: last 365 days only, a few calls per minute). A jump in circulating supply is new tokens
entering the market: unlocks to early investors / team, emissions, airdrops.

  event   circulating supply up >= 2% in a day (one event per coin per 14 days), coin has a Binance USDT perp
  test    the perp's return from 7 days before to 14 days after, minus the equal-weight return of all coins in the
          sample over the same days (market control); trade = short the day before an announced unlock? We cannot
          know CoinGecko's jump in advance, so the tradable version is short from the day after the jump for 14 days
  pass    after-jump excess return negative with t <= -2 (a first look: one year of data)

    python -m backend.unlock_lab collect      # ~45 minutes (rate limits)
    python -m backend.unlock_lab evaluate
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "unlocks"))
CG = "https://api.coingecko.com/api/v3"


def _get(url, params=None):
    for k in range(8):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(20 * (k + 1) if r.status_code == 429 else 5)
    return None


def collect(top: int = 220):
    from backend.universe_data import usdt_perps
    os.makedirs(DATA, exist_ok=True)
    perps = {s[:-4].replace("1000000", "").replace("1000", "").upper(): s for s in usdt_perps() if s.endswith("USDT")}
    mk_path = os.path.join(DATA, "markets.json")
    if not os.path.exists(mk_path):
        markets = []
        for page in range(1, 5):
            markets += _get(f"{CG}/coins/markets", {"vs_currency": "usd", "per_page": 250, "page": page}) or []
            time.sleep(15)
        json.dump(markets, open(mk_path, "w"))
    markets = json.load(open(mk_path))
    stable = {"usdt", "usdc", "dai", "fdusd", "tusd", "usde", "usds", "pyusd", "busd", "usd1"}
    picks, seen = [], set()
    for m in markets:
        sym = m["symbol"].upper()
        if sym in perps and sym not in seen and m["symbol"] not in stable:
            picks.append((m["id"], sym, perps[sym]))
            seen.add(sym)
    picks = picks[:top]
    print(f"{len(picks)} coins with a Binance perp", flush=True)
    for n, (cid, sym, perp) in enumerate(picks):
        path = os.path.join(DATA, f"{cid}.json")
        if os.path.exists(path):
            continue
        d = _get(f"{CG}/coins/{cid}/market_chart", {"vs_currency": "usd", "days": 365, "interval": "daily"})
        if d:
            json.dump({"sym": sym, "perp": perp, **d}, open(path, "w"))
        if n % 20 == 0:
            print(f"  {n}/{len(picks)}", flush=True)
        time.sleep(13)


def supply_panel():
    close, supply = {}, {}
    for f in os.listdir(DATA):
        if f == "markets.json" or not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(DATA, f)))
        if not d.get("prices"):
            continue
        p = pd.DataFrame(d["prices"], columns=["t", "p"]).set_index("t")["p"]
        m = pd.DataFrame(d["market_caps"], columns=["t", "m"]).set_index("t")["m"]
        idx = pd.to_datetime(p.index, unit="ms").normalize()
        p.index, m.index = idx, pd.to_datetime(m.index, unit="ms").normalize()
        p, m = p[~p.index.duplicated()], m[~m.index.duplicated()]
        close[d["perp"]] = p
        supply[d["perp"]] = (m / p).where(m > 0)
    return pd.DataFrame(close).sort_index(), pd.DataFrame(supply).sort_index()


def evaluate():
    close, supply = supply_panel()
    ret = close.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    jump = supply.pct_change(fill_method=None)
    rows = []
    for sym in close.columns:
        last = None
        for day in jump.index[(jump[sym] >= 0.02) & (jump[sym] < 1.0)]:
            if last is not None and (day - last).days < 14:
                continue
            last = day
            i = close.index.get_loc(day)
            if i < 8 or i + 15 >= len(close):
                continue
            ex = (ret[sym] - mkt).iloc

            def cum(a, b):
                return float((1 + ex[i + a:i + b + 1]).prod() - 1)
            rows.append({"sym": sym, "day": day, "jump": jump[sym].iloc[i], "before": cum(-7, -1), "day0": cum(0, 0),
                         "after": cum(1, 14)})
    ev = pd.DataFrame(rows)

    def t(x):
        x = x.dropna()
        return f"{100 * x.mean():+.2f}% (t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}, n {len(x)}, median {100 * x.median():+.2f}%)"
    print(f"{close.shape[1]} coins, {len(ev)} supply jumps >= 2% ({ev.day.min():%Y-%m-%d} .. {ev.day.max():%Y-%m-%d})")
    print(f"  excess return vs all coins: 7 days before {t(ev.before)} | jump day {t(ev.day0)} | next 14 days {t(ev.after)}")
    big = ev[ev.jump >= 0.05]
    print(f"  jumps >= 5% ({len(big)}): before {t(big.before)} | after {t(big.after)}")
    print(f"  tradable: short the day after the jump for 14 days, 0.2% costs: {t(-ev.after - 0.002)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    collect() if cmd == "collect" else evaluate()
