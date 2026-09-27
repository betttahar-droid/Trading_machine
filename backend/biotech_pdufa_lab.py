"""
The FDA decision-date run-up: do drug / biotech stocks rise into a published PDUFA date (the FDA's target date to
approve or reject a drug), so that buying weeks before and selling the day before earns without taking the
approval gamble?

Data: SEC 8-K exhibits mentioning "PDUFA" (2015-01 .. now, drug / biotech SIC codes), plus the press releases already
downloaded by biotech_lab / biotech_start_lab. The PDUFA date is read from the text ("PDUFA target action date of
March 25, 2024"); the first filing that names a (company, date) pair is when the date became public.

Rule fixed before looking at results:
  buy at the close 40 trading days before the PDUFA date (only if the date was public by then), sell at the close
  one trading day before it; abnormal return vs XBI; 0.5% round-trip costs; pass = positive net in 2015-21 and 2022+
  Control: the same stock over the 40 days ending 3 months before the entry.

    python -m backend.biotech_pdufa_lab collect
    python -m backend.biotech_pdufa_lab evaluate
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

DATA = os.path.join(os.path.dirname(bl.DATA), "biotech_pdufa")
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(rf"PDUFA[^.;]{{0,200}}?\b({MONTHS})\s+(\d{{1,2}}),\s+(20\d\d)", re.I)
HOLD, CONTROL_GAP = 40, 60


def collect():
    os.makedirs(DATA, exist_ok=True)
    hits_path = os.path.join(DATA, "hits.csv")
    if os.path.exists(hits_path):
        hits = pd.read_csv(hits_path)
    else:
        with ThreadPoolExecutor(2) as ex:
            parts = list(ex.map(bl._search_one, ['"PDUFA"', '"target action date"']))
        hits = pd.DataFrame([r for p in parts for r in p])
        hits.to_csv(hits_path, index=False)
    hits["pr"] = hits["file_type"].fillna("").str.upper().eq("EX-99.1")
    docs = hits.sort_values(["adsh", "pr"], ascending=[True, False]).drop_duplicates("adsh").reset_index(drop=True)
    print(f"{len(hits)} matching exhibits -> {len(docs)} filings from {docs.cik.nunique()} companies")
    have = {}
    for path in (os.path.join(bl.DATA, "docs.jsonl"), os.path.join(os.path.dirname(bl.DATA), "biotech_start", "docs.jsonl")):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                d = json.loads(line)
                have[d["adsh"]] = d
    out = os.path.join(DATA, "docs.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(line)["adsh"] for line in open(out, encoding="utf-8")}
    todo = docs[~docs.adsh.isin(done)]
    with open(out, "a", encoding="utf-8") as f:
        reuse = todo[todo.adsh.isin(have)]
        for r in reuse.itertuples():
            f.write(json.dumps(have[r.adsh]) + "\n")
        todo = todo[~todo.adsh.isin(have)]
        print(f"  reused {len(reuse)}, downloading {len(todo)}", flush=True)
        with ThreadPoolExecutor(4) as ex:
            for n, (row, text) in enumerate(zip(todo.itertuples(), ex.map(bl._text, todo.itertuples()))):
                f.write(json.dumps({"adsh": row.adsh, "file_date": row.file_date, "cik": row.cik, "name": row.name,
                                    "tickers": row.tickers, "sic": row.sic, "text": text}) + "\n")
                if n % 250 == 0:
                    f.flush()
                    print(f"  downloaded {n}/{len(todo)}", flush=True)
                time.sleep(0.1)


def pdufa_events() -> pd.DataFrame:
    rows = []
    for line in open(os.path.join(DATA, "docs.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        filed = pd.Timestamp(d["file_date"])
        for m in DATE_RE.finditer(d["text"]):
            try:
                day = pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}")
            except ValueError:
                continue
            if filed < day <= filed + pd.Timedelta(days=730):
                rows.append({"cik": d["cik"], "ticker": bl._ticker(d["tickers"]), "name": d["name"],
                             "pdufa": day, "public": filed})
    ev = pd.DataFrame(rows).sort_values("public")
    return ev.groupby(["cik", "pdufa"], as_index=False).first()          # first filing naming the date


def evaluate():
    ev = pdufa_events()
    p = pd.read_pickle(os.path.join(bl.DATA, "prices.pkl"))
    close, xbi = p["close"], p["close"]["XBI"].dropna()
    cal = xbi.index
    need = sorted({t for t in ev.ticker if t} - set(close.columns))
    if need:
        from backend.biotech_start_lab import update_prices
        update_prices(ev.assign(ticker=ev.ticker))
        p = pd.read_pickle(os.path.join(bl.DATA, "prices.pkl"))
        close = p["close"]
    rows = []
    for r in ev.itertuples():
        if r.ticker not in close.columns:
            rows.append({"has_price": False})
            continue
        c = close[r.ticker].reindex(cal)
        iP = cal.searchsorted(r.pdufa)                   # PDUFA day (or next trading day)
        ex, en = iP - 1, iP - 1 - HOLD
        if en - CONTROL_GAP - HOLD < 0 or iP >= len(cal) or cal[en] < r.public:
            rows.append({"has_price": False, "too_late": en >= 0 and iP < len(cal) and cal[max(en, 0)] < r.public})
            continue

        def ar(a, b):
            x = c.iloc[b] / c.iloc[a] - 1
            return x - (xbi.iloc[b] / xbi.iloc[a] - 1) if np.isfinite(x) else np.nan
        rows.append({"has_price": np.isfinite(c.iloc[en]) and np.isfinite(c.iloc[ex]),
                     "runup": ar(en, ex), "last20": ar(ex - 20, ex),
                     "control": ar(en - CONTROL_GAP - HOLD, en - CONTROL_GAP),
                     "decision_day": ar(ex, min(iP + 1, len(cal) - 1))})
    ev = pd.concat([ev.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    ev["period"] = np.where(ev["pdufa"] < "2022-01-01", "2015-21", "2022+")
    print(f"{len(ev)} (company, PDUFA date) pairs; {ev.has_price.fillna(False).mean():.0%} tradable with prices "
          f"(date public 40+ trading days ahead)")
    e = ev[ev.has_price.fillna(False).astype(bool)]
    for per in ("2015-21", "2022+"):
        g = e[e.period == per]
        print(f"\n  {per}: run-up (40 days to the day before) {bl._t(g.runup)}")
        print(f"          net of 0.5% costs {bl._t(g.runup - 0.005)} | median {100 * g.runup.median():+.1f}%")
        print(f"          last 20 days only {bl._t(g.last20)} | control window {bl._t(g.control)}")
        print(f"          (not traded) decision day {bl._t(g.decision_day)}")
    # calendar-time portfolio
    rx = xbi.pct_change()
    legs = []
    for r in e.itertuples():
        iP = cal.searchsorted(r.pdufa)
        seg = slice(iP - HOLD, iP)
        legs.append((close[r.ticker].reindex(cal).pct_change().iloc[seg] - rx.iloc[seg]).fillna(0.0))
    pos = pd.concat(legs, axis=1)
    n = pos.notna().sum(axis=1).reindex(cal).fillna(0)
    port = (pos.sum(axis=1).reindex(cal).fillna(0) / n.replace(0, np.nan)).fillna(0) - (0.005 / HOLD) * (n > 0)
    for lab, part in (("2015-21", port["2015":"2021"]), ("2022+", port["2022":])):
        eq = (1 + part).cumprod()
        print(f"  portfolio {lab}: {(eq.iloc[-1] ** (252 / len(part)) - 1):+.1%}/yr, Sharpe "
              f"{part.mean() / part.std() * np.sqrt(252):+.2f}, max DD {(eq / eq.cummax() - 1).min():+.0%}, "
              f"invested {(n.reindex(part.index) > 0).mean():.0%} of days")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if cmd == "collect":
        collect()
    else:
        evaluate()
