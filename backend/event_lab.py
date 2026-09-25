"""
Exchange announcements: can a small, reasonably fast bot trade them?

Events come from the exchanges' official Telegram channels (telegram_data.py, timestamps to the second):
  @binance_announcements   "Binance Will List X (TICKER)" -> long,  "Binance Will Delist A, B" -> short
  @upbit_news              KRW/BTC/USDT market additions (디지털 자산 추가) -> long,
                           caution designations (유의 종목 지정) and end of trading support (거래지원 종료) -> short
  @bithumb_notice          market additions (마켓 추가) -> long, caution / delisting -> short

Each event is traded on the token's Binance USD-M perpetual when one already exists, with 1-minute klines
(data.binance.vision daily files). Entry at the open of the minute after the post (0-60 s late, "fast") or two
minutes after (60-120 s, "slow"); exits after 5, 15, 60, 240 and 1440 minutes. Costs: taker fee 0.05% and
slippage 0.3% per side (fast, thin markets) = 0.7% per round trip. "pre-move" is how far price had already moved
between the post's minute open and the fast entry.

    python -m backend.event_lab
"""

import io
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from backend.backtest_trend import DATA_DIR
from backend.telegram_data import load_channel
from backend.universe_data import load_funding, usdt_perps

CACHE = os.path.join(DATA_DIR, "event_cache")
COST = 2 * (0.0005 + 0.003)
HORIZONS = (5, 15, 60, 240, 1440)


def _tickers(text: str) -> List[str]:
    return [t for t in re.findall(r"\(([A-Z0-9]{2,12})\)", text) if t not in ("USDT", "USDC", "KRW", "BTC", "UTC", "KST")]


def binance_events() -> List[dict]:
    out = []
    for m in load_channel("binance_announcements"):
        first = m["text"].split("\n")[0]
        if re.search(r"Binance Will List\b", first):
            out += [{"ts": m["ts"], "src": "binance", "kind": "list", "side": 1, "ticker": t, "text": first} for t in _tickers(first)]
        elif re.search(r"Binance Will Delist\b", first):
            names = re.sub(r".*Will Delist\s+", "", first).split(" on ")[0]
            for t in re.split(r",\s*|\s+and\s+|\s*&\s*", names):
                t = t.strip().upper()
                if re.fullmatch(r"[A-Z0-9]{2,12}", t):
                    out.append({"ts": m["ts"], "src": "binance", "kind": "delist", "side": -1, "ticker": t, "text": first})
    return out


def korean_events(channel: str, src: str) -> List[dict]:
    out = []
    for m in load_channel(channel):
        text = m["text"].replace("\n", " ")
        if src == "upbit" and not text.startswith("[거래]"):   # only trading notices, not event promos
            continue
        if "해제" in text or "안내 (완료)" in text or "기념" in text or "이벤트" in text:
            continue                                   # caution released, follow-ups, promotions
        if re.search(r"(디지털 자산 추가|마켓 추가|신규 상장|원화 마켓 상장)", text):
            kind, side = "list", 1
        elif re.search(r"유의 종목 지정|투자유의종목 지정|유의 촉구", text):
            kind, side = "caution", -1
        elif re.search(r"거래지원 종료|거래 지원 종료|상장 폐지", text):
            kind, side = "delist", -1
        else:
            continue
        for t in _tickers(text):
            if any(e["ticker"] == t and e["kind"] == kind and m["ts"] - e["ts"] < 7 * 86400 for e in out[-50:]):
                continue                               # same notice re-posted
            out.append({"ts": m["ts"], "src": src, "kind": kind, "side": side, "ticker": t, "text": text[:120]})
    return out


def minute_bars(symbol: str, day: str) -> Optional[pd.DataFrame]:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{symbol}-1m-{day}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0) if os.path.getsize(path) > 10 else None
    url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{day}.zip"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=30)
            break
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    else:
        return None
    if r.status_code != 200:
        open(path, "w").close()
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode().splitlines()
    rows = [line.split(",")[:5] for line in raw if line[:1].isdigit()]
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"]).astype(float)
    df["t"] = (df["t"] // 60_000).astype(int)          # minute number
    df = df.set_index("t")
    df.to_csv(path)
    return df


def trade(ev: dict, perps: set) -> Optional[dict]:
    sym = ev["ticker"] + "USDT"
    if sym not in perps:
        sym = "1000" + ev["ticker"] + "USDT"
        if sym not in perps:
            return None
    k0 = ev["ts"] // 60
    days = sorted({datetime.fromtimestamp(k * 60, tz=timezone.utc).strftime("%Y-%m-%d") for k in (k0, k0 + 1500)})
    frames = [f for f in (minute_bars(sym, d) for d in days) if f is not None]
    if not frames:
        return None
    bars = pd.concat(frames)
    bars = bars[~bars.index.duplicated()]
    if k0 not in bars.index or k0 - 60 not in bars.index:       # perp must already trade before the post
        return None
    out = {**ev, "symbol": sym, "pre_move": ev["side"] * (bars.at[k0 + 1, "open"] / bars.at[k0, "open"] - 1)
           if k0 + 1 in bars.index else np.nan}
    for lag, name in ((1, "fast"), (2, "slow")):
        if k0 + lag not in bars.index:
            return None
        entry = bars.at[k0 + lag, "open"]
        for h in HORIZONS:
            k = k0 + lag + h - 1
            out[f"{name}_{h}"] = ev["side"] * (bars.at[k, "close"] / entry - 1) - COST if k in bars.index else np.nan
    # Slow entry, 4h and 24h holds: funding paid/received, worst move against the trade, and stop-losses
    entry, k_in = bars.at[k0 + 2, "open"], k0 + 2
    months = sorted({datetime.fromtimestamp(k * 60, tz=timezone.utc).strftime("%Y-%m") for k in (k_in, k_in + 1440)})
    fund = load_funding(sym, months)
    for h in (240, 1440):
        win = bars.loc[k_in:k_in + h - 1]
        if len(win) < h * 0.9:
            continue
        adverse = (win["high"].max() / entry - 1) if ev["side"] < 0 else (1 - win["low"].min() / entry)
        out[f"mae_{h}"] = adverse
        t0, t1 = pd.Timestamp(k_in * 60, unit="s"), pd.Timestamp((k_in + h) * 60, unit="s")
        f = fund[(fund.index > t0) & (fund.index <= t1)].sum() if len(fund) else 0.0
        out[f"fund_{h}"] = -ev["side"] * f                    # longs pay positive funding, shorts receive it
        out[f"slow_{h}_f"] = out[f"slow_{h}"] + out[f"fund_{h}"]
        for stop in (0.10, 0.20):
            hit = win["high"] >= entry * (1 + stop) if ev["side"] < 0 else win["low"] <= entry * (1 - stop)
            if hit.any():
                k = hit.idxmax()
                gap = (win.at[k, "open"] / entry - 1) * -ev["side"] if ev["side"] < 0 else (1 - win.at[k, "open"] / entry)
                out[f"slow_{h}_stop{int(stop * 100)}"] = -max(stop, gap) - COST + out[f"fund_{h}"]
            else:
                out[f"slow_{h}_stop{int(stop * 100)}"] = out[f"slow_{h}_f"]
    return out


def summary(df: pd.DataFrame, label: str):
    if df.empty:
        print(f"  {label}: no tradable events")
        return
    print(f"  {label}: n={len(df)}, {pd.to_datetime(df['ts'], unit='s').min():%Y-%m} .. "
          f"{pd.to_datetime(df['ts'], unit='s').max():%Y-%m}, median pre-move before a fast entry {df['pre_move'].median():+.1%}")
    for name in ("fast", "slow"):
        cells = []
        for h in HORIZONS:
            x = df[f"{name}_{h}"].dropna()
            cells.append(f"{h}m {x.mean():+5.1%}/{x.median():+5.1%} ({(x > 0).mean():3.0%})")
        print(f"    {name:<4} mean/median (win%) after costs: " + " | ".join(cells))


def detail(df: pd.DataFrame, label: str):
    """Slow entry (60-120 s late), after costs: plain, with funding, with 10% / 20% stops; tails and halves."""
    print(f"\n  {label}: n={len(df)}")
    for h in (240, 1440):
        cols = [(f"slow_{h}", "no funding"), (f"slow_{h}_f", "with funding"),
                (f"slow_{h}_stop20", "20% stop"), (f"slow_{h}_stop10", "10% stop")]
        for c, name in cols:
            x = df[c].dropna()
            if x.empty:
                continue
            first, second = x.iloc[:len(x) // 2], x.iloc[len(x) // 2:]
            print(f"    {h:>4}m {name:<12} mean {x.mean():+6.1%} median {x.median():+6.1%} win {(x > 0).mean():4.0%} "
                  f"worst {x.min():+7.1%} 10th pct {x.quantile(0.1):+6.1%} | first half {first.mean():+6.1%}, "
                  f"second half {second.mean():+6.1%}")
        mae = df[f"mae_{h}"].dropna()
        print(f"    {h:>4}m worst move against the trade: median {mae.median():+.1%}, 90th pct {mae.quantile(0.9):+.1%}, "
              f"max {mae.max():+.1%}")


def main():
    perps = set(usdt_perps())
    events = binance_events() + korean_events("upbit_news", "upbit") + korean_events("bithumb_notice", "bithumb")
    # Fading the listing pump: the same listing events traded short
    events += [{**e, "kind": "list_fade", "side": -1} for e in events if e["kind"] == "list"]
    print(f"{len(events)} announcement events parsed", flush=True)
    with ThreadPoolExecutor(16) as ex:
        rows = [r for r in ex.map(lambda e: trade(e, perps), events) if r]
    df = pd.DataFrame(rows)
    os.makedirs(CACHE, exist_ok=True)
    df.to_csv(os.path.join(CACHE, "event_trades.csv"), index=False)
    print(f"{len(df)} events had a Binance perp already trading\n")
    df = df.sort_values("ts")
    print("\nDetail for the two candidates:")
    detail(df[(df["src"] == "upbit") & (df["kind"] == "list_fade")], "Upbit listing, fade (short the pump)")
    detail(df[(df["src"] == "binance") & (df["kind"] == "delist")], "Binance delisting, short")
    detail(df[(df["src"] == "binance") & (df["kind"] == "list_fade")], "Binance listing, fade (short)")
    detail(df[(df["src"] == "bithumb") & (df["kind"] == "list")], "Bithumb KRW listing, long")
    print("\nAll event types:")
    for (src, kind), g in df.groupby(["src", "kind"]):
        summary(g, f"{src} {kind} ({'long' if g['side'].iloc[0] > 0 else 'short'})")
        if src == "binance":
            for label, part in (("before 2024-07", g[g["ts"] < 1719792000]), ("2024-07 on", g[g["ts"] >= 1719792000])):
                summary(part, f"   {label}")


if __name__ == "__main__":
    main()
