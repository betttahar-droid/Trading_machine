"""
Backtest of the news safety brake (news_guard.py) on 2020-2024 headlines.

Uses the CoinDesk/CryptoCompare news archive on Hugging Face (maryamfakhari/crypto-news-coindesk-2020-2025,
~229k headlines from ~40 crypto outlets, CC BY-NC 4.0, research use). Laya takes ~1 s per headline on a
CPU, so only headlines containing an event word (hack, withdraw, delist, sue, depeg, outage, ...) and
either naming a universe coin or a market-wide trigger word are classified. Verdicts are cached in
data/news_archive/verdicts.jsonl, so runs can be stopped and resumed.

    python -m backend.news_guard_backtest            # headlines in the 24h before each trend entry
    python -m backend.news_guard_backtest --all      # every candidate headline + event study

Default mode answers "would the brake have skipped losing trades?": blocked entries' R vs. the rest,
and vs. blocking the same number of entries at random. --all also measures what the blocked coins did
in the 1-7 days after each flagged headline, compared with any other time.
"""

import json
import os
import re
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
import requests

from backend.backtest_trend import DATA_DIR, SPLIT, _ms
from backend.growth_study import LIVE, WINDOW_START, _date, load_market_bulk, symbol_trades
from backend.news_guard import ALIASES, BLOCK_HOURS, EVENT_GATE, KEYWORDS, NewsGuard
from backend.trend_strategy import compute_features

ARCHIVE_URL = ("https://huggingface.co/datasets/maryamfakhari/crypto-news-coindesk-2020-2025/"
               "resolve/main/coindesk-crypto-news-2020-2025.csv")
ARCHIVE_DIR = os.path.join(DATA_DIR, "news_archive")
ARCHIVE = os.path.join(ARCHIVE_DIR, "coindesk-2020-2025.csv")
VERDICTS = os.path.join(ARCHIVE_DIR, "verdicts.jsonl")
EVENT_WORDS = sorted({w for ws in KEYWORDS.values() for w in ws} | {
    "hack", "exploit", "drain", "breach", "withdraw", "insolven", "bankrupt", "halt", "paus", "freez", "frozen",
    "delist", "sue", "charge", "lawsuit", "sanction", "ban", "fine", "depeg", "peg", "outage", "stall", "offline",
    "arrest", "seize", "collapse", "attack", "stolen", "theft", "suspend"})
H4 = 4 * 3600


def load_headlines() -> List[dict]:
    import pandas as pd
    if not os.path.exists(ARCHIVE):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        print(f"Downloading news archive (~240 MB) to {ARCHIVE} ...", flush=True)
        with requests.get(ARCHIVE_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(ARCHIVE, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    df = pd.read_csv(ARCHIVE, usecols=["published_on", "title"]).dropna()
    df["ts"] = (pd.to_datetime(df["published_on"]) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    df["key"] = df["title"].str.strip().str.lower()
    df = df.sort_values("ts").drop_duplicates("key")       # first publication of each title
    warnings.filterwarnings("ignore", "This pattern is interpreted as a regular expression")
    ev = df["key"].str.contains("|".join(re.escape(w) for w in EVENT_WORDS))
    coin = df["key"].str.contains("|".join(rf"\b{re.escape(a)}\b" for al in ALIASES.values() for a in al))
    gate = df["key"].str.contains("|".join(f"(?:{EVENT_GATE[e]})" for e in ("depeg", "exchange_failure")))
    df = df[ev & (coin | gate)]
    return [{"ts": int(t), "title": s} for t, s in zip(df["ts"], df["title"])]


def classify(guard: NewsGuard, items: List[dict]) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if os.path.exists(VERDICTS):
        with open(VERDICTS, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["title"]] = rec
    todo = [it for it in items if it["title"] not in cache]
    if todo:
        print(f"Classifying {len(todo)} headlines with Laya (~{len(todo) / 60:.0f} min on CPU) ...", flush=True)
    t0 = time.time()
    with open(VERDICTS, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo):
            rec = {"title": it["title"], **guard.evaluate(it["title"])}
            cache[it["title"]] = rec
            f.write(json.dumps(rec) + "\n")
            if n % 250 == 249:
                f.flush()
                print(f"  ... {n + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return cache


def main():
    import laya
    all_mode = "--all" in sys.argv
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    market = load_market_bulk("4h", last_month)
    heads = load_headlines()
    end_news = heads[-1]["ts"]
    print(f"{len(heads)} candidate headlines {_date(heads[0]['ts'] * 1000)} .. {_date(end_news * 1000)}")

    trades = []
    for sym, m in market.items():
        for t in symbol_trades(m, compute_features(m["bars"], LIVE), LIVE, _ms(WINDOW_START)):
            entry_s = t["ts"] // 1000 + H4           # filled at the open after the signal bar
            if entry_s < end_news:
                trades.append({**t, "sym": sym, "entry_s": entry_s})
    print(f"{len(trades)} trend entries before the archive ends")

    window = BLOCK_HOURS * 3600
    ts_arr = np.array([h["ts"] for h in heads])

    def before(entry_s):
        lo, hi = np.searchsorted(ts_arr, entry_s - window), np.searchsorted(ts_arr, entry_s)
        return heads[lo:hi]

    need = heads if all_mode else list({h["title"]: h for t in trades for h in before(t["entry_s"])}.values())
    guard = NewsGuard(universe=list(market))
    guard.attach_agent(laya.load("convaiinnovations/laya", device="cpu"))
    verdicts = classify(guard, need)

    # --- 1. entries the brake would have skipped
    blocked, reasons = [], {}
    for k, t in enumerate(trades):
        for h in before(t["entry_s"]):
            v = verdicts.get(h["title"])
            if v and v["flagged"] and (v["scope"] == "market" or t["sym"] in v["scope"]):
                blocked.append(k)
                reasons[k] = f"{v['event']} ({v['prob']:.2f}): {h['title'][:80]}"
                break
    r = np.array([t["r"] for t in trades])
    mask = np.zeros(len(r), bool)
    mask[blocked] = True
    print(f"\nBrake would have skipped {mask.sum()} of {len(r)} entries:")
    for k in blocked:
        print(f"  {_date(trades[k]['entry_s'] * 1000)} {trades[k]['sym']:<9} R {r[k]:+6.2f}  {reasons[k]}")
    if mask.any():
        rng = np.random.default_rng(0)
        rand = np.array([r[rng.choice(len(r), mask.sum(), replace=False)].mean() for _ in range(10_000)])
        print(f"  skipped entries E[R] {r[mask].mean():+.3f} (total {r[mask].sum():+.1f}R) vs kept {r[~mask].mean():+.3f}")
        print(f"  random skips of the same size were at least this bad {np.mean(rand <= r[mask].mean()):.0%} of the time "
              f"(small = the brake picked genuinely bad entries)")

    if not all_mode:
        return

    # --- 2. event study: blocked coins' returns after flagged headlines vs. any time
    closes = {s: {b["timestamp"] // 1000: b["close"] for b in m["bars"]} for s, m in market.items()}
    horizons = (1, 3, 7)
    flagged = [(h, verdicts[h["title"]]) for h in heads if verdicts.get(h["title"], {}).get("flagged")]
    out = {d: [] for d in horizons}
    by_event: Dict[str, List[float]] = {}
    for h, v in flagged:
        t = (h["ts"] // H4 + 1) * H4                   # first 4h close after publication
        syms = list(market) if v["scope"] == "market" else v["scope"]
        for d in horizons:
            vals = [closes[s][t + d * 86400] / closes[s][t] - 1 for s in syms
                    if t in closes[s] and t + d * 86400 in closes[s]]
            if vals:
                out[d].append(np.mean(vals))
                if d == 3:
                    by_event.setdefault(v["event"], []).append(np.mean(vals))
    base = {}
    for d in horizons:
        vals = [c[t + d * 86400] / c[t] - 1 for c in closes.values() for t in c
                if _ms(WINDOW_START) // 1000 <= t < end_news and t + d * 86400 in c]
        base[d] = float(np.mean(vals))
    print(f"\nEvent study: {len(flagged)} flagged headlines (2020-2024)")
    for d in horizons:
        a = np.array(out[d])
        print(f"  +{d}d: after flagged {a.mean():+.2%} (median {np.median(a):+.2%}, n={len(a)}) | any time {base[d]:+.2%}")
    print("  +3d by event type: " + ", ".join(f"{e} {np.mean(x):+.2%} (n={len(x)})" for e, x in sorted(by_event.items())))


if __name__ == "__main__":
    main()
