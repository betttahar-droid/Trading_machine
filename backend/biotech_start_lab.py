"""
Buying at the start of a breakthrough: FDA designations and research-start announcements by drug / biotech companies.

  BTD       Breakthrough Therapy Designation (the FDA judged early clinical data a substantial improvement)
  FAST      Fast Track designation          ORPHAN   Orphan Drug Designation      RPD   Rare Pediatric Disease
  RMAT      Regenerative Medicine Advanced Therapy designation                   PRIO  Priority Review
  IND       FDA clearance of an Investigational New Drug application (first tests in humans may start)
  FPD       first patient dosed             P3START  a Phase 3 trial started

Same data and returns as backend/biotech_lab.py (SEC 8-K exhibits 2015-01 .. now, the event must be in the headline
or lede, Yahoo prices, abnormal return vs XBI, entry at the close of the day after the filing).

Rules fixed before looking at results:
  main    long after a Breakthrough Therapy Designation, 60 trading days, beats XBI after 0.5% costs in 2015-21
          and in 2022+
  others  the same for every other event type (reported, weaker evidence because many are tested)
  control the same stocks' abnormal return over the 60 days ending 3 months before the event: surviving companies
          may beat XBI anyway, so the event effect is the difference

    python -m backend.biotech_start_lab collect
    python -m backend.biotech_start_lab evaluate
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from backend import biotech_lab as bl

DATA = os.path.join(os.path.dirname(bl.DATA), "biotech_start")
PHRASES = {"BTD": ['"Breakthrough Therapy Designation"'], "FAST": ['"Fast Track designation"'],
           "ORPHAN": ['"Orphan Drug Designation"'], "PRIO": ['"Priority Review"'],
           "RPD": ['"Rare Pediatric Disease"'], "RMAT": ['"RMAT designation"'],
           "IND": ['"clearance of its Investigational New Drug"', '"cleared the Investigational New Drug"',
                   '"IND clearance"'],
           "FPD": ['"first patient dosed"'], "P3START": ['"initiation of a Phase 3"', '"initiated a Phase 3"']}
GRANT = r"(?:grant|receiv|award|obtain)"
LEDE = {"BTD": rf"breakthrough therapy designation.{{0,200}}|{GRANT}\w*.{{0,120}}breakthrough therapy",
        "FAST": rf"{GRANT}\w*.{{0,120}}fast track|fast track (?:designation|status).{{0,80}}{GRANT}",
        "ORPHAN": rf"{GRANT}\w*.{{0,120}}orphan drug|orphan drug designation.{{0,80}}{GRANT}",
        "PRIO": r"priority review",
        "RPD": rf"{GRANT}\w*.{{0,120}}rare pediatric|rare pediatric disease designation.{{0,80}}{GRANT}",
        "RMAT": r"rmat designation|regenerative medicine advanced therapy",
        "IND": r"(?:clear\w*|clearance).{0,80}(?:\bIND\b|investigational new drug)|(?:\bIND\b|investigational new drug).{0,80}clear",
        "FPD": r"first patient (?:dosed|treated)",
        "P3START": r"(?:initiat|start|commence|launch|first patient).{0,60}phase (?:3|iii)\b"}


def collect():
    os.makedirs(DATA, exist_ok=True)
    hits_path = os.path.join(DATA, "hits.csv")
    if os.path.exists(hits_path):
        hits = pd.read_csv(hits_path)
    else:
        jobs = [(k, p) for k, ps in PHRASES.items() for p in ps]
        with ThreadPoolExecutor(3) as ex:
            parts = list(ex.map(lambda kp: [dict(r, kind=kp[0]) for r in bl._search_one(kp[1])], jobs))
        hits = pd.DataFrame([r for p in parts for r in p])
        hits.to_csv(hits_path, index=False)
    hits["pr"] = hits["file_type"].fillna("").str.upper().eq("EX-99.1")
    docs = hits.sort_values(["adsh", "pr"], ascending=[True, False]).drop_duplicates("adsh").reset_index(drop=True)
    print(f"{len(hits)} matching exhibits -> {len(docs)} filings from {docs.cik.nunique()} companies")
    out = os.path.join(DATA, "docs.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(line)["adsh"] for line in open(out, encoding="utf-8")}
    # reuse press releases already downloaded by biotech_lab
    old = {}
    old_path = os.path.join(bl.DATA, "docs.jsonl")
    if os.path.exists(old_path):
        for line in open(old_path, encoding="utf-8"):
            d = json.loads(line)
            old[d["adsh"]] = d
    todo = docs[~docs.adsh.isin(done)]
    with open(out, "a", encoding="utf-8") as f:
        reuse = todo[todo.adsh.isin(old)]
        for r in reuse.itertuples():
            f.write(json.dumps(old[r.adsh]) + "\n")
        todo = todo[~todo.adsh.isin(old)]
        print(f"  reused {len(reuse)} press releases, downloading {len(todo)}", flush=True)
        with ThreadPoolExecutor(4) as ex:
            for n, (row, text) in enumerate(zip(todo.itertuples(), ex.map(bl._text, todo.itertuples()))):
                f.write(json.dumps({"adsh": row.adsh, "file_date": row.file_date, "cik": row.cik, "name": row.name,
                                    "tickers": row.tickers, "sic": row.sic, "text": text}) + "\n")
                if n % 250 == 0:
                    f.flush()
                    print(f"  downloaded {n}/{len(todo)}", flush=True)
                time.sleep(0.1)


def events() -> pd.DataFrame:
    docs = pd.read_json(os.path.join(DATA, "docs.jsonl"), lines=True).drop_duplicates("adsh")
    docs["text"] = docs["text"].map(bl._clean)
    docs = docs[~docs["text"].str[:400].str.contains(bl.EARNINGS)]
    docs["ticker"] = docs["tickers"].map(bl._ticker)
    lede = docs["text"].str[:700]
    rows = []
    for kind, pat in LEDE.items():
        hit = docs[lede.str.contains(pat, case=False, regex=True)]
        rows.append(hit.assign(kind=kind))
    ev = pd.concat(rows)
    ev["file_date"] = pd.to_datetime(ev["file_date"])
    ev = ev.sort_values("file_date").reset_index(drop=True)
    keep, last = [], {}
    for r in ev.itertuples():                   # one event per company and type per 30 days
        key = (r.cik, r.kind)
        if key in last and (r.file_date - last[key]).days < 30:
            continue
        last[key] = r.file_date
        keep.append(r.Index)
    return ev.loc[keep].reset_index(drop=True)


def update_prices(ev: pd.DataFrame):
    """Add Yahoo prices for tickers not yet in biotech_lab's price cache."""
    import yfinance as yf
    path = os.path.join(bl.DATA, "prices.pkl")
    p = pd.read_pickle(path)
    need = sorted({t for t in ev["ticker"] if t} - set(p["close"].columns))
    close, vol = {}, {}
    for i in range(0, len(need), 80):
        d = yf.download(need[i:i + 80], start="2014-06-01", progress=False, auto_adjust=True, threads=True)
        for t in need[i:i + 80]:
            try:
                c = d["Close"][t].dropna()
            except KeyError:
                continue
            if len(c):
                close[t], vol[t] = c, d["Volume"][t].reindex(c.index)
    if close:
        p = {"close": pd.concat([p["close"], pd.DataFrame(close)], axis=1),
             "volume": pd.concat([p["volume"], pd.DataFrame(vol)], axis=1)}
        pd.to_pickle(p, path)
    print(f"prices: {len(close)} of {len(need)} new tickers found")


def evaluate():
    ev = events()
    update_prices(ev)
    ev = bl.returns(ev)
    p = pd.read_pickle(os.path.join(bl.DATA, "prices.pkl"))
    close, xbi = p["close"], p["close"]["XBI"].dropna()
    cal = xbi.index

    def placebo(r):
        if not r.has_price:
            return np.nan
        c = close[r.ticker].reindex(cal)
        i0 = cal.searchsorted(r.file_date)
        a, b = i0 - 121, i0 - 61
        if a < 0 or not np.isfinite(c.iloc[a]) or not np.isfinite(c.iloc[b]):
            return np.nan
        return (c.iloc[b] / c.iloc[a] - 1) - (xbi.iloc[b] / xbi.iloc[a] - 1)
    ev["placebo60"] = [placebo(r) for r in ev.itertuples()]
    ev["period"] = np.where(ev["file_date"] < "2022-01-01", "2015-21", "2022+")
    print(f"{len(ev)} events, {ev.has_price.mean():.0%} with prices")
    print("  events by type: " + ", ".join(f"{k} {n}" for k, n in ev["kind"].value_counts().items()))
    e = ev[ev.has_price]
    for kind in LEDE:
        print(f"\n  {kind}")
        for per in ("2015-21", "2022+"):
            g = e[(e.kind == kind) & (e.period == per)]
            net = g["post60"] - 0.005
            print(f"    {per}: reaction {bl._t(g.react)} | next 60d {bl._t(g.post60)} | control {bl._t(g.placebo60)} | "
                  f"60d minus control {100 * (g.post60.mean() - g.placebo60.mean()):+.2f}% | net of costs {bl._t(net)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if cmd == "collect":
        collect()
    else:
        evaluate()
