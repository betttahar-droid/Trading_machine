"""
Drug-trial results and biotech stocks: do the shares keep drifting after a trial readout, and can a small local model
(one that fits a 4 GB RTX 3050) read the announcements well enough to trade it?

Data (all free):
  events   SEC EDGAR full-text search: 8-K filings 2015-01 .. now by drug / biotech companies (SIC 2833-2836, 8731)
           whose exhibits mention trial readouts ("topline results", "met its primary endpoint", ...). The press
           release exhibit is downloaded; the SEC asks automated tools to identify themselves with a contact address.
  prices   Yahoo Finance daily closes (companies that were later delisted are usually missing: survivorship, see below)
  bench    XBI (S&P biotech ETF): abnormal return = stock return - XBI return

Trading rule and tests are in evaluate(); the classification runs on a GPU with backend/news_llm_score.py-style
letter log-probabilities (biotech_score.py).

    python -m backend.biotech_lab collect     # search + download press releases -> data/biotech/
    python -m backend.biotech_lab prices
    python -m backend.biotech_lab evaluate
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "biotech"))
UA = {"User-Agent": "Trading_machine research betttahar@gmail.com"}
PHRASES = ['"topline results"', '"top-line results"', '"topline data"', '"top-line data"',
           '"met its primary endpoint"', '"met the primary endpoint"', '"did not meet its primary endpoint"',
           '"did not meet the primary endpoint"', '"primary efficacy endpoint"']
SICS = {"2833", "2834", "2835", "2836", "8731"}
START, END = "2015-01-01", time.strftime("%Y-%m-%d")


def _get(url: str, params=None, tries: int = 5):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                time.sleep(5 * (k + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(3 * (k + 1))
    return None


def _search_one(phrase: str) -> list:
    rows = []
    for year in range(int(START[:4]), int(END[:4]) + 1):
        frm = 0
        while True:
            r = _get("https://efts.sec.gov/LATEST/search-index",
                     {"q": phrase, "forms": "8-K", "dateRange": "custom", "startdt": f"{year}-01-01",
                      "enddt": f"{year}-12-31", "from": frm})
            time.sleep(0.3)
            if r is None:
                break
            hits = r.json()["hits"]
            for h in hits["hits"]:
                s = h["_source"]
                if not SICS & set(s.get("sics") or []):
                    continue
                name = (s.get("display_names") or [""])[0]
                m = re.search(r"\(([A-Z0-9\.\-, ]+)\)\s+\(CIK", name)
                rows.append({"adsh": s["adsh"], "file": h["_id"].split(":", 1)[1], "file_type": s.get("file_type"),
                             "file_date": s["file_date"], "cik": s["ciks"][0], "name": name.split("  (")[0],
                             "tickers": m.group(1) if m else "", "sic": (s.get("sics") or [""])[0], "phrase": phrase})
            frm += 100
            if frm >= min(hits["total"]["value"], 10_000):
                break
    print(f"  {phrase}: {len(rows)} drug/biotech exhibits", flush=True)
    return rows


def search() -> pd.DataFrame:
    """All matching 8-K exhibits, per phrase and year (the search returns at most 10,000 hits per query)."""
    with ThreadPoolExecutor(3) as ex:                     # ~10 requests/s is the SEC's limit
        parts = list(ex.map(_search_one, PHRASES))
    return pd.DataFrame([r for p in parts for r in p])


def _text(row) -> str:
    url = f"https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/{row.adsh.replace('-', '')}/{row.file}"
    r = _get(url)
    if r is None:
        return ""
    from bs4 import BeautifulSoup
    t = BeautifulSoup(r.content, "lxml").get_text(" ")
    return re.sub(r"\s+", " ", t).strip()[:6000]


def collect():
    os.makedirs(DATA, exist_ok=True)
    hits_path = os.path.join(DATA, "hits.csv")
    hits = pd.read_csv(hits_path) if os.path.exists(hits_path) else search()
    hits.to_csv(hits_path, index=False)
    # one document per filing: the press release (EX-99.1) if it matched, else the first matching exhibit
    hits["pr"] = hits["file_type"].fillna("").str.upper().eq("EX-99.1")
    docs = (hits.sort_values(["adsh", "pr"], ascending=[True, False]).drop_duplicates("adsh")
            .reset_index(drop=True))
    print(f"{len(hits)} matching exhibits -> {len(docs)} filings from {docs.cik.nunique()} companies")
    out = os.path.join(DATA, "docs.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(line)["adsh"] for line in open(out, encoding="utf-8")}
    todo = docs[~docs.adsh.isin(done)]
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(4) as ex:
        for n, (row, text) in enumerate(zip(todo.itertuples(), ex.map(_text, todo.itertuples()))):
            f.write(json.dumps({"adsh": row.adsh, "file_date": row.file_date, "cik": row.cik, "name": row.name,
                                "tickers": row.tickers, "sic": row.sic, "text": text}) + "\n")
            if n % 250 == 0:
                f.flush()
                print(f"  downloaded {n}/{len(todo)}", flush=True)
            time.sleep(0.1)


EARNINGS = re.compile(r"financial results|quarter(?:ly)? (?:\d{4} )?(?:financial )?results|(?:full|fiscal)[- ]year (?:\d{4} )?results"
                      r"|annual results", re.I)


def _clean(t: str) -> str:
    """Skip the 8-K cover page when the document is the filing itself rather than the press release."""
    head = t[:4000]
    m = re.search(r"Item\s+(?:8\.01|7\.01)", head)
    if m and "FORM 8-K" in head[:2000].upper():
        return t[m.start():]
    return t


def readouts() -> pd.DataFrame:
    """Downloaded filings minus earnings releases (quarterly / annual results with a pipeline update)."""
    docs = pd.read_json(os.path.join(DATA, "docs.jsonl"), lines=True)
    docs["text"] = docs["text"].map(_clean)
    return docs[~docs["text"].str[:400].str.contains(EARNINGS)].reset_index(drop=True)


def write_input():
    d = readouts()
    d[["adsh", "file_date", "name", "text"]].assign(text=d["text"].str[:3000]).to_json(
        os.path.join(DATA, "input.jsonl"), orient="records", lines=True, force_ascii=False)
    print(f"{len(d)} filings to classify (earnings releases removed)")


def _ticker(s) -> str:
    return str(s or "").split(",")[0].strip()


def prices():
    import yfinance as yf
    docs = pd.read_json(os.path.join(DATA, "docs.jsonl"), lines=True)
    tickers = sorted({_ticker(t) for t in docs["tickers"] if _ticker(t)} | {"XBI"})
    close, vol = {}, {}
    for i in range(0, len(tickers), 80):
        batch = tickers[i:i + 80]
        d = yf.download(batch, start="2014-06-01", progress=False, auto_adjust=True, threads=True)
        for t in batch:
            try:
                c = d["Close"][t].dropna()
            except KeyError:
                continue
            if len(c):
                close[t], vol[t] = c, d["Volume"][t].reindex(c.index)
        print(f"  prices {i + len(batch)}/{len(tickers)}: {len(close)} found", flush=True)
    pd.to_pickle({"close": pd.DataFrame(close), "volume": pd.DataFrame(vol)}, os.path.join(DATA, "prices.pkl"))


POS = re.compile(r"met (?:its|the|both|all) (?:\w+ )?(?:primary|co-primary) (?:efficacy )?endpoints?|achieved (?:its|the) primary"
                 r"|statistically significant (?:improvement|reduction|benefit)", re.I)
NEG = re.compile(r"did not (?:meet|achieve)|failed to (?:meet|achieve|demonstrate)|not statistically significant"
                 r"|discontinu\w+ (?:the |its )?(?:trial|study|development|program)", re.I)


def events(model: str) -> pd.DataFrame:
    docs = readouts()
    docs["ticker"] = docs["tickers"].map(_ticker)
    if model == "keywords":
        head = docs["text"].str[:2500]
        pos, neg = head.str.contains(POS), head.str.contains(NEG)
        docs["outcome"] = np.select([pos & ~neg, neg & ~pos, pos & neg], ["positive", "negative", "mixed"], "none")
        docs["phase"] = np.where(head.str.contains(r"phase (?:3|iii)|pivotal", case=False), "3", "?")
        ev = docs
    else:
        sc = pd.read_csv(os.path.join(DATA, f"bio_{model}.csv"))
        ev = docs.merge(sc, on="adsh")
        best = ev[["out_A", "out_B", "out_C"]].idxmax(axis=1).map({"out_A": "positive", "out_B": "mixed", "out_C": "negative"})
        ev["outcome"] = np.where(ev["out_D"] >= 0.5, "none", best)
        ev["phase"] = ev[["phase_A", "phase_B", "phase_C", "phase_D"]].idxmax(axis=1).str[-1].map(
            {"A": "1", "B": "2", "C": "3", "D": "?"})
    ev = ev[ev["outcome"] != "none"].copy()
    ev["file_date"] = pd.to_datetime(ev["file_date"])
    ev = ev.sort_values("file_date")
    keep, last = [], {}
    for r in ev.itertuples():                   # one readout per company per 30 days (re-filed / follow-up releases)
        key = r.cik
        if key in last and (r.file_date - last[key]).days < 30:
            continue
        last[key] = r.file_date
        keep.append(r.Index)
    return ev.loc[keep].reset_index(drop=True)


def returns(ev: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_pickle(os.path.join(DATA, "prices.pkl"))
    close, volume = p["close"], p["volume"]
    xbi = close["XBI"].dropna()
    cal = xbi.index
    rows = []
    for r in ev.itertuples():
        base = {"has_price": False}
        if r.ticker in close.columns:
            c = close[r.ticker].reindex(cal)
            i0 = cal.searchsorted(r.file_date)                 # the filing day (or next trading day)
            if 62 <= i0 and i0 + 62 < len(cal) and np.isfinite(c.iloc[[i0 - 2, i0 + 1]]).all():
                def ar(a, b):
                    x = c.iloc[b] / c.iloc[a] - 1
                    return x - (xbi.iloc[b] / xbi.iloc[a] - 1) if np.isfinite(x) else np.nan
                dv = (c * volume[r.ticker].reindex(cal)).iloc[i0 - 62:i0 - 2].mean()
                # if the stock stops trading inside the window, use its last price (delisting / merger)
                last = c.iloc[i0 + 1:i0 + 62].last_valid_index()
                j20 = min(i0 + 21, cal.get_loc(last)) if last is not None else i0 + 1
                j60 = min(i0 + 61, cal.get_loc(last)) if last is not None else i0 + 1
                base = {"has_price": True, "react": ar(i0 - 2, i0 + 1), "post20": ar(i0 + 1, j20),
                        "post60": ar(i0 + 1, j60), "dollar_vol": dv}
        rows.append(base)
    return pd.concat([ev.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _t(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) < 5:
        return f"n={len(x)}"
    return f"{100 * x.mean():+6.2f}% (t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}, n {len(x)}, {100 * (x > 0).mean():.0f}% up)"


def evaluate(models):
    for model in models:
        ev = returns(events(model))
        ev["period"] = np.where(ev["file_date"] < "2022-01-01", "2015-21", "2022+")
        print(f"\n################ {model}: {len(ev)} readouts, {ev.has_price.mean():.0%} with prices ################")
        print("  share with prices by outcome: " + ", ".join(f"{o}: {g.has_price.mean():.0%} of {len(g)}"
                                                          for o, g in ev.groupby("outcome")))
        ev = ev[ev.has_price]
        big = ev["dollar_vol"] >= ev.loc[ev.period == "2015-21", "dollar_vol"].median()
        ev["size"] = np.where(big, "large", "small")
        for per in ("2015-21", "2022+"):
            e = ev[ev.period == per]
            print(f"\n  {per}   (abnormal returns vs XBI; reaction = 2 days before filing .. day after)")
            for o in ("positive", "mixed", "negative"):
                g = e[e.outcome == o]
                print(f"    {o:8s} reaction {_t(g.react)} | next 20d {_t(g.post20)} | next 60d {_t(g.post60)}")
            for o in ("positive", "negative"):
                for s in ("small", "large"):
                    g = e[(e.outcome == o) & (e["size"] == s)]
                    print(f"    {o:8s} {s:5s} reaction {_t(g.react)} | next 60d {_t(g.post60)}")
            p3 = e[e.phase == "3"]
            for o in ("positive", "negative"):
                g = p3[p3.outcome == o]
                print(f"    phase 3 {o:8s} reaction {_t(g.react)} | next 60d {_t(g.post60)}")
            cost = 0.005
            h1 = pd.concat([e.loc[e.outcome == "positive", "post60"], -e.loc[e.outcome == "negative", "post60"]]) - cost
            neg = e[e.outcome == "negative"]
            h2 = pd.concat([-neg.loc[neg["size"] == "large", "post60"], neg.loc[neg["size"] == "small", "post60"]]) - cost
            print(f"    RULE 1 long positive / short negative, 60d, 0.5% costs: {_t(h1)}")
            print(f"    RULE 2 short large-firm failures / buy small-firm failures, 60d: {_t(h2)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if cmd == "collect":
        collect()
    elif cmd == "prices":
        prices()
    elif cmd == "input":
        write_input()
    else:
        evaluate(sys.argv[2:] or ["keywords"])
