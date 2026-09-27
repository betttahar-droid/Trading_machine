"""
Can a large language model read breaking news and call Bitcoin's next move?

Every @WatcherGuru Telegram post (2022-02 .. now; price ticks and ads removed) is shown to an instruction-tuned LLM,
which answers with one letter, A (strongly down) .. G (strongly up), for Bitcoin over the next few hours. The score
is the probability-weighted answer (-3 .. +3) from the letter log-probabilities (backend/news_llm_score.py, run on a
rented GPU with vLLM). Only posts after the model's training data ends are a fair test: Qwen2.5 (released 2024-09)
cannot know what happened after 2024, so 2025-01 .. now is the test window and earlier years only show how much a
model "predicts" news it has already seen.

Rules fixed before looking at any result:
  entry   BTCUSDT perp, open of the first full minute at least 60 s after the post (1-2 minutes late)
  trade   long if score >= +1, short if <= -1; hold 15m / 1h / 4h / 24h (1h is the main horizon); one position at
          a time; 0.15% round-trip costs (taker fee + slippage)
  pass    mean net return > 0 in 2025 and in 2026, p < 0.01 over the test window (random-direction permutation),
          and better than the naive rule "follow the last 15 minutes' move" on the same posts

    python -m backend.news_llm_lab prepare     # posts -> data/news_llm/posts.jsonl, BTC 1-minute prices
    python -m backend.news_llm_lab evaluate    # data/news_llm/scores_*.csv -> report
"""

import calendar
import glob
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUT = os.path.join(DATA, "news_llm")
POSTS = os.path.join(OUT, "posts.jsonl")
PRICES = os.path.join(OUT, "btc_1m_open.npz")
TEST_START = pd.Timestamp("2025-01-01").timestamp()
HORIZONS = {"15m": 15, "1h": 60, "4h": 240, "24h": 1440}
COST = 0.0015
THRESH = 1.0

_TICK = re.compile(r"^\s*(🟢|🔴|🟩|🟥)?\s*\$[\d,\.]+\s*(@\w+)?\s*$")


def clean(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@WatcherGuru\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_posts():
    os.makedirs(OUT, exist_ok=True)
    rows, seen = [], set()
    for line in open(os.path.join(DATA, "telegram", "WatcherGuru.jsonl"), encoding="utf-8"):
        m = json.loads(line)
        t = clean(m.get("text", ""))
        if len(t) < 25 or _TICK.match(t) or "@bitcoin_price" in m.get("text", "") or "DROP A LIKE" in t.upper():
            continue
        if (t, m["ts"] // 3600) in seen:
            continue
        seen.add((t, m["ts"] // 3600))
        rows.append({"id": m["id"], "ts": int(m["ts"]), "text": t[:600]})
    rows.sort(key=lambda r: r["ts"])
    with open(POSTS, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    test = sum(r["ts"] >= TEST_START for r in rows)
    print(f"{len(rows)} posts kept ({test} in the 2025+ test window) -> {POSTS}")


def _zip_rows(url: str):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            break
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    else:
        return []
    if r.status_code != 200:
        return []
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return [line.split(",")[:2] for line in z.read(z.namelist()[0]).decode().splitlines() if line[:1].isdigit()]


def prepare_prices(start="2022-02", sym="BTCUSDT"):
    base = f"https://data.binance.vision/data/futures/um"
    rows = []
    months = pd.period_range(start, pd.Timestamp.utcnow().strftime("%Y-%m"), freq="M")
    for p in months:
        got = _zip_rows(f"{base}/monthly/klines/{sym}/1m/{sym}-1m-{p}.zip")
        if not got:                                      # current month: daily files
            for d in pd.date_range(p.start_time, min(p.end_time, pd.Timestamp.utcnow().tz_localize(None)), freq="D"):
                got += _zip_rows(f"{base}/daily/klines/{sym}/1m/{sym}-1m-{d:%Y-%m-%d}.zip")
        rows += got
        print(f"  {p}: {len(got)} minutes", flush=True)
    t = np.array([int(r[0]) // 60_000 for r in rows]); o = np.array([float(r[1]) for r in rows])
    order = np.argsort(t); t, o = t[order], o[order]
    full = np.arange(t[0], t[-1] + 1)
    px = pd.Series(o, index=t).groupby(level=0).first().reindex(full).ffill().to_numpy()
    np.savez_compressed(PRICES, t0=full[0], open=px)
    print(f"{len(full)} minutes {pd.Timestamp(full[0] * 60, unit='s')} .. {pd.Timestamp(full[-1] * 60, unit='s')}")


def load_prices():
    z = np.load(PRICES)
    return int(z["t0"]), z["open"]


def outcomes(posts: pd.DataFrame) -> pd.DataFrame:
    t0, px = load_prices()
    m0 = (posts["ts"].to_numpy() + 60 + 59) // 60 - t0      # first full minute >= 60 s after the post
    ok = (m0 >= 15) & (m0 + max(HORIZONS.values()) < len(px))
    posts = posts[ok].copy(); m0 = m0[ok]
    posts["entry_min"] = m0
    posts["pre15"] = px[m0] / px[m0 - 15] - 1
    for k, h in HORIZONS.items():
        posts[f"r_{k}"] = px[m0 + h] / px[m0] - 1
    return posts


def trades(df: pd.DataFrame, direction: np.ndarray, h: int, hk: str) -> pd.DataFrame:
    """Non-overlapping trades in time order; direction 0 = no trade."""
    out, free_at = [], -1
    for i, d in zip(range(len(df)), direction):
        if d == 0 or df["entry_min"].iat[i] < free_at:
            continue
        free_at = df["entry_min"].iat[i] + h
        out.append((df.index[i], d, d * df[f"r_{hk}"].iat[i] - COST))
    return pd.DataFrame(out, columns=["idx", "dir", "net"]).set_index("idx")


def perm_p(df, direction, h, hk, n=1000, seed=0):
    """p-value: random directions on the same entry times."""
    base = trades(df, direction, h, hk)
    if len(base) < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    gross = df.loc[base.index, f"r_{hk}"].to_numpy()
    obs = base["net"].mean()
    sims = [(rng.choice([-1, 1], size=len(gross)) * gross - COST).mean() for _ in range(n)]
    return (np.sum(np.array(sims) >= obs) + 1) / (n + 1)


def report(df: pd.DataFrame, label: str):
    print(f"\n=== {label}: {len(df)} posts; score vs the move before entry (15 min): "
          f"{df['score'].corr(df['pre15'], method='spearman'):+.3f} ===")
    rows = []
    llm_dir = np.where(df["score"] >= THRESH, 1, np.where(df["score"] <= -THRESH, -1, 0))
    mom_dir = np.where(llm_dir != 0, np.sign(df["pre15"].to_numpy()), 0).astype(int)
    for hk, h in HORIZONS.items():
        rho = df["score"].corr(df[f"r_{hk}"], method="spearman")
        t = trades(df, llm_dir, h, hk)
        m = trades(df, mom_dir, h, hk)
        yr = pd.to_datetime(df.loc[t.index, "ts"], unit="s").dt.year
        by_year = t["net"].groupby(yr.to_numpy()).mean()
        rows.append({"horizon": hk, "spearman": rho, "trades": len(t), "long%": (t["dir"] > 0).mean() if len(t) else np.nan,
                     "net/trade %": 100 * t["net"].mean() if len(t) else np.nan,
                     "win %": 100 * (t["net"] > 0).mean() if len(t) else np.nan,
                     "p(random)": perm_p(df, llm_dir, h, hk),
                     "follow-move net %": 100 * m["net"].mean() if len(m) else np.nan,
                     **{f"{y} net %": 100 * v for y, v in by_year.items()}})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def evaluate():
    posts = pd.read_json(POSTS, lines=True)
    base = outcomes(posts)
    for f in sorted(glob.glob(os.path.join(OUT, "scores_*.csv"))):
        s = pd.read_csv(f)
        model = os.path.basename(f)[7:-4]
        df = base.merge(s[["id", "score", "impact"]], on="id").sort_values("ts").reset_index(drop=True)
        print(f"\n################ {model} ################")
        print(f"score distribution (test window): {df[df.ts >= TEST_START]['score'].describe().round(2).to_dict()}")
        report(df[df.ts >= TEST_START].reset_index(drop=True), f"{model} TEST 2025-01 .. now (model cannot know these)")
        report(df[df.ts < TEST_START].reset_index(drop=True), f"{model} 2022-02 .. 2024-12 (inside training data)")


def decode_log(path: str):
    """Rebuild scores_<name>.csv from the '[b64 <name> <i>]' lines news_llm_job.py prints into the instance log."""
    import base64
    import gzip
    parts: dict = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\[b64 (\w+) (\d+)\] (\S+)", line)
        if m:
            parts.setdefault(m.group(1), {})[int(m.group(2))] = m.group(3)
    for name, chunks in parts.items():
        blob = "".join(chunks[i] for i in sorted(chunks))
        with open(os.path.join(OUT, f"scores_{name}.csv"), "wb") as f:
            f.write(gzip.decompress(base64.b64decode(blob)))
        print(f"scores_{name}.csv rebuilt from {len(chunks)} log lines")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if cmd == "prepare":
        prepare_posts()
        prepare_prices()
    elif cmd == "decode":
        decode_log(sys.argv[2])
    else:
        evaluate()
