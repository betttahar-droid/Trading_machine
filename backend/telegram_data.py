"""
History of public Telegram channels from their web preview (https://t.me/s/<channel>), no login or API key.

Each page shows ~20 messages; `?before=<id>` pages back through the whole history. Messages are cached
in data/telegram/<channel>.jsonl ({"id", "ts", "text"}), so runs resume and later runs only fetch
what is new. Requests are paced (~1/s) and back off when Telegram rate-limits.

    python -m backend.telegram_data WatcherGuru whale_alert_io

parse_whale() turns Whale Alert posts ("613 $BTC (51,921,889 USD) transferred from Coinbase to unknown
wallet") into coin / USD / direction relative to exchanges.
"""

import html
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

from backend.backtest_trend import DATA_DIR

TG_DIR = os.path.join(DATA_DIR, "telegram")
UA = {"User-Agent": "Mozilla/5.0 (research; github.com/betttahar-droid/Trading_machine)"}
MSG_RE = re.compile(r'data-post="[^/"]+/(\d+)".*?<time datetime="([^"]+)"', re.S)
TEXT_RE = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)


def _page(channel: str, before: Optional[int] = None) -> List[dict]:
    url = f"https://t.me/s/{channel}" + (f"?before={before}" if before else "")
    status = None
    for attempt in range(8):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            status = r.status_code
            if status == 200:
                break
        except requests.RequestException as e:          # dropped connections happen on long runs
            status = repr(e)
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"t.me/s/{channel} failed: {status}")
    out = []
    for block in r.text.split('<div class="tgme_widget_message_wrap')[1:]:
        m = MSG_RE.search(block)
        if not m:
            continue
        t = TEXT_RE.search(block)
        text = re.sub(r"<br\s*/?>", "\n", t.group(1)) if t else ""
        text = html.unescape(re.sub(r"<[^>]+>", "", text)).strip()
        ts = int(datetime.fromisoformat(m.group(2)).timestamp())
        out.append({"id": int(m.group(1)), "ts": ts, "text": text})
    return out


def load_channel(channel: str) -> List[dict]:
    path = os.path.join(TG_DIR, f"{channel}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        msgs = {m["id"]: m for m in map(json.loads, f)}
    return sorted(msgs.values(), key=lambda m: m["id"])


def fetch_channel(channel: str, pause: float = 1.0) -> List[dict]:
    """Fetch new messages (newest first) until reaching cached ones, then continue into older history."""
    os.makedirs(TG_DIR, exist_ok=True)
    path = os.path.join(TG_DIR, f"{channel}.jsonl")
    have = {m["id"] for m in load_channel(channel)}
    oldest = min(have) if have else None
    before, done_new, n_new, t0 = None, False, 0, time.time()
    with open(path, "a", encoding="utf-8") as f:
        while True:
            page = _page(channel, before)
            fresh = [m for m in page if m["id"] not in have]
            for m in fresh:
                f.write(json.dumps(m) + "\n")
                have.add(m["id"])
            n_new += len(fresh)
            if not page:
                break
            low = min(m["id"] for m in page)
            if not done_new and not fresh and oldest is not None:
                done_new, before = True, oldest          # caught up with the cache: jump to its oldest end
            else:
                before = low
            if before <= 1:
                break
            if n_new and n_new % 1000 < len(fresh):
                f.flush()
                print(f"  {channel}: {n_new} new messages, back to {datetime.utcfromtimestamp(page[0]['ts']):%Y-%m-%d} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            time.sleep(pause)
    msgs = load_channel(channel)
    print(f"{channel}: {len(msgs)} messages cached ({n_new} new)", flush=True)
    return msgs


EXCHANGES = ("binance", "coinbase", "kraken", "okx", "okex", "bybit", "bitfinex", "huobi", "htx", "kucoin", "gemini",
             "bitstamp", "bithumb", "upbit", "gate.io", "bitget", "crypto.com", "mexc", "ftx", "poloniex", "bittrex")
WHALE_RE = re.compile(r"([\d,\.]+)\s+[#$]?([A-Z0-9]+)\s+\(([\d,\.]+)\s+USD\)\s+transferred from\s+(.+?)\s+to\s+(.+?)(?:\s+Details|$)", re.S)


def _is_exchange(party: str) -> bool:
    p = party.lower()
    return any(e in p for e in EXCHANGES)


def parse_whale(msg: dict) -> Optional[dict]:
    """{'coin', 'usd', 'flow'} with flow = +1 into an exchange, -1 out of one, 0 otherwise; None if not a transfer."""
    m = WHALE_RE.search(msg["text"].replace("\n", " "))
    if not m:
        return None
    src, dst = m.group(4), m.group(5)
    flow = (1 if _is_exchange(dst) else 0) - (1 if _is_exchange(src) else 0)
    return {"ts": msg["ts"], "coin": m.group(2).upper(), "usd": float(m.group(3).replace(",", "")), "flow": flow}


def main():
    for ch in sys.argv[1:] or ["WatcherGuru", "whale_alert_io"]:
        fetch_channel(ch)


if __name__ == "__main__":
    main()
