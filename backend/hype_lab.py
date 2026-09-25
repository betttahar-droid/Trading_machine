"""
Does news tone, hype or public attention help the trend strategy?

For every entry of the live trend strategy (8 coins, 4h, 120-bar breakout) this measures, at entry time:

  news_count    headlines naming the coin in the 72h before entry (CoinDesk/CryptoCompare archive, ~40
                outlets, 2020 .. 2025-01; see news_guard_backtest.py)
  news_attn     news_count relative to the coin's prior 30-day headline rate (an attention spike)
  laya_tone     Laya's P(bullish) - P(bearish) averaged over up to 30 of those headlines
  laya_hype     Laya's P(headline is hype: price predictions, "to the moon", FOMO), averaged
  wiki_attn     English Wikipedia pageviews of the coin's article over the 3 days before entry relative to
                the prior 30 days (public attention; available up to now, so it has a real out-of-sample)

Each measure is fixed in advance (no thresholds fitted): trades are split into thirds by the measure and
the top third is compared with the bottom third, with a permutation test (how often a random split
gives a gap at least as large). X/Twitter history is not available without a paid API, and Reddit's
API refuses these requests, so news and Wikipedia stand in for "social hype".

    python -m backend.hype_lab
"""

import json
import os
import re
import time
import zlib
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

from backend.backtest_trend import DATA_DIR, SPLIT, UNIVERSE, _ms
from backend.growth_study import LIVE, WINDOW_START, load_market_bulk, symbol_trades
from backend.news_guard import ALIASES
from backend.news_guard_backtest import ARCHIVE, load_headlines
from backend.trend_strategy import compute_features
from backend.universe_data import available_months

CACHE = os.path.join(DATA_DIR, "hype_cache")
TONE_CACHE = os.path.join(CACHE, "laya_tone.jsonl")
WIKI_ARTICLES = {
    "BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "SOLUSDT": "Solana_(blockchain_platform)",
    "BNBUSDT": "Binance", "XRPUSDT": "XRP_Ledger", "DOGEUSDT": "Dogecoin",
    "AVAXUSDT": "Avalanche_(blockchain_platform)", "SUIUSDT": None,
}
PER_TRADE = 30          # headlines Laya reads per trade (sampled deterministically)
TONE_Q = {
    "impact": {"type": "choice", "instructions": "What does this headline mean for the price of the cryptocurrency it is about?",
               "criteria": {"bullish": "good for the price: adoption, partnerships, inflows, upgrades, approvals, rising price, buying",
                            "bearish": "bad for the price: hacks, lawsuits, bans, outflows, delays, selling, falling price",
                            "neutral": "no clear effect: explainers, how-to guides, opinions without direction, unrelated events"}},
    "hype": {"type": "noul", "instructions": "Is this headline hype: a bold price prediction, 'to the moon' rally talk, or fear of missing out?"},
}
H4, DAY = 4 * 3600, 86400


def all_headlines() -> pd.DataFrame:
    """Every archive headline (not only event candidates), deduplicated, with the coins it names."""
    if not os.path.exists(ARCHIVE):
        load_headlines()                                 # downloads the archive
    df = pd.read_csv(ARCHIVE, usecols=["published_on", "title"]).dropna()
    df["ts"] = (pd.to_datetime(df["published_on"]) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    df["key"] = df["title"].str.strip().str.lower()
    df = df.sort_values("ts").drop_duplicates("key").reset_index(drop=True)
    for sym, aliases in ALIASES.items():
        df[sym] = df["key"].str.contains("|".join(rf"\b{re.escape(a)}\b" for a in aliases))
    return df


def wiki_views(article: str, start: str = "20200101") -> pd.Series:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"wiki_{article}.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
        items = json.load(open(path))
    else:
        end = time.strftime("%Y%m%d")
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
               f"{article}/daily/{start}/{end}")
        for attempt in range(8):
            r = requests.get(url, headers={"User-Agent": "TradingMachineResearch/1.0 (github.com/betttahar-droid/Trading_machine)"},
                             timeout=30)
            if r.status_code == 200:
                items = r.json()["items"]
                break
            time.sleep(15 * (attempt + 1))       # Wikimedia rate limit
        else:
            raise RuntimeError(f"Wikipedia pageviews unavailable for {article}: {r.status_code}")
        json.dump(items, open(path, "w"))
    s = pd.Series({pd.Timestamp(x["timestamp"][:8]): x["views"] for x in items}, dtype=float)
    return s.sort_index()


def laya_tone(titles: List[str]) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if os.path.exists(TONE_CACHE):
        for line in open(TONE_CACHE, encoding="utf-8"):
            rec = json.loads(line)
            cache[rec["title"]] = rec
    todo = [t for t in dict.fromkeys(titles) if t not in cache]
    if todo:
        import laya
        os.makedirs(CACHE, exist_ok=True)
        agent = laya.load("convaiinnovations/laya", device="cpu")
        print(f"Laya reading {len(todo)} headlines (~{len(todo) * 0.5 / 60:.0f} min on CPU) ...", flush=True)
        t0 = time.time()
        with open(TONE_CACHE, "a", encoding="utf-8") as f:
            for n, t in enumerate(todo):
                a = agent.system_one(t, TONE_Q)["answers"]
                p = a["impact"]["probabilities"]
                rec = {"title": t, "bull": p["bullish"], "bear": p["bearish"], "hype": a["hype"]["noul"]}
                cache[t] = rec
                f.write(json.dumps(rec) + "\n")
                if n % 250 == 249:
                    f.flush()
                    print(f"  ... {n + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return cache


def split_test(name: str, x: np.ndarray, r: np.ndarray, rng) -> str:
    ok = ~np.isnan(x)
    x, r = x[ok], r[ok]
    if len(x) < 30:
        return f"  {name:<11} n={len(x)}: too few"
    lo, hi = np.quantile(x, [1 / 3, 2 / 3])
    top, bot = r[x >= hi], r[x <= lo]
    gap = top.mean() - bot.mean()
    perm = []
    for _ in range(5000):
        s = rng.permutation(r)
        perm.append(s[:len(top)].mean() - s[len(top):len(top) + len(bot)].mean())
    p = np.mean(np.abs(perm) >= abs(gap))
    return (f"  {name:<11} n={len(x):3d}  top third E[R] {top.mean():+.2f} | middle {r[(x > lo) & (x < hi)].mean():+.2f} | "
            f"bottom third {bot.mean():+.2f}  -> gap {gap:+.2f}, random split this large {p:.0%}")


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    market = load_market_bulk("4h", last_month, UNIVERSE, lambda s: available_months(s, "4h"))
    trades = []
    for sym, m in market.items():
        for t in symbol_trades(m, compute_features(m["bars"], LIVE), LIVE, _ms(WINDOW_START)):
            trades.append({"sym": sym, "entry_s": t["ts"] // 1000 + H4, "r": t["r"]})
    trades = pd.DataFrame(trades).sort_values("entry_s").reset_index(drop=True)
    print(f"{len(trades)} trend trades on the 8 coins", flush=True)

    heads = all_headlines()
    news_end = int(heads["ts"].max())
    ts = heads["ts"].to_numpy()
    sample, feats = {}, []
    for k, t in trades.iterrows():
        row = {"news_count": np.nan, "news_attn": np.nan}
        if t.entry_s < news_end:
            lo72, lo30 = np.searchsorted(ts, t.entry_s - 3 * DAY), np.searchsorted(ts, t.entry_s - 33 * DAY)
            hi = np.searchsorted(ts, t.entry_s)
            coin = heads[t.sym].to_numpy()
            recent = np.flatnonzero(coin[lo72:hi]) + lo72
            before = coin[lo30:lo72].sum()
            row["news_count"] = len(recent)
            row["news_attn"] = len(recent) / max(before / 10.0, 1.0)
            titles = sorted(heads["title"].to_numpy()[recent], key=lambda s: zlib.crc32(s.encode()))[:PER_TRADE]
            sample[k] = titles
        feats.append(row)
    trades = pd.concat([trades, pd.DataFrame(feats)], axis=1)

    tone = laya_tone([t for titles in sample.values() for t in titles])
    trades["laya_tone"] = [np.mean([tone[t]["bull"] - tone[t]["bear"] for t in sample[k]]) if sample.get(k) else np.nan
                           for k in trades.index]
    trades["laya_hype"] = [np.mean([tone[t]["hype"] for t in sample[k]]) if sample.get(k) else np.nan
                           for k in trades.index]

    wiki = {}
    for sym, art in WIKI_ARTICLES.items():
        if art:
            wiki[sym] = wiki_views(art)
            time.sleep(3)
    vals = []
    for t in trades.itertuples():
        s = wiki.get(t.sym)
        day = pd.Timestamp(t.entry_s, unit="s").normalize()
        if s is None:
            vals.append(np.nan)
            continue
        w3 = s[day - pd.Timedelta(days=3):day - pd.Timedelta(days=1)].mean()
        w30 = s[day - pd.Timedelta(days=33):day - pd.Timedelta(days=4)].mean()
        vals.append(w3 / w30 if w30 and not np.isnan(w30) else np.nan)
    trades["wiki_attn"] = vals

    rng = np.random.default_rng(0)
    r = trades["r"].to_numpy()
    oos = (trades["entry_s"] >= _ms(SPLIT) // 1000).to_numpy()
    print(f"\nAll trades E[R] {r.mean():+.2f}. Split by each measure at entry (top vs bottom third):")
    for name in ("news_count", "news_attn", "laya_tone", "laya_hype", "wiki_attn"):
        print(split_test(name, trades[name].to_numpy(), r, rng))
    print(f"\nOut-of-sample only ({SPLIT} on; news ends {pd.Timestamp(news_end, unit='s'):%Y-%m}):")
    for name in ("news_attn", "laya_tone", "laya_hype", "wiki_attn"):
        print(split_test(name, trades[name].to_numpy()[oos], r[oos], rng))
    print("\nCorrelation between measures:")
    print(trades[["news_attn", "laya_tone", "laya_hype", "wiki_attn"]].corr(method="spearman").round(2).to_string())
    os.makedirs(CACHE, exist_ok=True)
    trades.to_csv(os.path.join(CACHE, "trend_trades_hype.csv"), index=False)


if __name__ == "__main__":
    main()
