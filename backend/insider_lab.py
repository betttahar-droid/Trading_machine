"""
Insider buying across all US stocks. Open-market purchases by officers and directors are among the best-documented
stock signals (Lakonishok & Lee 2001; Jeng, Metrick & Zeckhauser 2003); Cohen, Malloy & Pomorski (2012) found the
return comes from "opportunistic" buys, not from insiders who buy in the same month every year. The biotech-only
version decayed (biotech_insider_lab.py); this is the full market, which a retail investor can trade with a broker.

Data: SEC Insider Transactions Data Sets (Form 4, 2008 .. 2026Q1); prices from Yahoo. A signal's ticker must have a
Yahoo close within 20% of the insider's own purchase price on the trade date (split-corrected), which drops reused
tickers and bad matches. Delisted companies are missing from Yahoo; the share of signals without prices is reported.

Fixed in advance:
  signal   Form 4 with officer / director open-market purchases (code P) >= $25,000 in total; one per company per
           30 days; entry at the close of the first trading day after the filing date
  groups   all | CEO or CFO buyer | cluster (3+ different insiders in 30 days) | opportunistic (the insider did not
           buy in the same calendar month in each of the 3 previous years)
  return   60 trading days, minus IWM (small caps) over the same days; 0.5% round-trip costs
  periods  2010-2018 (design) vs 2019-2026
  pass     net abnormal return > 0 in both periods with t >= 2 in 2019+, and a calendar-time portfolio with
           Sharpe > 0.5 in 2019+

    python -m backend.insider_lab collect
    python -m backend.insider_lab prices
    python -m backend.insider_lab evaluate
"""

import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from backend.biotech_lab import _get

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "insider"))
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
MIN_VALUE, HOLD, COST, SPLIT = 25_000, 60, 0.005, "2019-01-01"


def collect():
    os.makedirs(DATA, exist_ok=True)
    out = []
    quarters = [f"{y}q{q}" for y in range(2008, int(time.strftime("%Y")) + 1) for q in range(1, 5)]
    for q in quarters:
        path = os.path.join(DATA, f"{q}.csv")
        if os.path.exists(path):
            out.append(pd.read_csv(path, dtype=str))
            continue
        r = _get(URL.format(q=q))
        if r is None:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = {n.split("/")[-1].upper(): n for n in z.namelist()}

        def tab(name, cols):
            return pd.read_csv(z.open(names[name]), sep="\t", usecols=cols, dtype=str, on_bad_lines="skip")
        sub = tab("SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "DOCUMENT_TYPE"])
        sub = sub[sub["DOCUMENT_TYPE"].isin(["4", "4/A"])]
        tr = tab("NONDERIV_TRANS.TSV", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                         "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
        tr = tr[(tr["TRANS_CODE"] == "P") & (tr["TRANS_ACQUIRED_DISP_CD"] == "A")]
        own = tab("REPORTINGOWNER.TSV", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
        m = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER")
        m = m[["ACCESSION_NUMBER", "FILING_DATE", "TRANS_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "RPTOWNERCIK",
               "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"]]
        m.to_csv(path, index=False)
        out.append(m)
        print(f"  {q}: {len(m)} purchase rows", flush=True)
        time.sleep(0.3)
    pd.concat(out).to_csv(os.path.join(DATA, "purchases.csv"), index=False)


def purchases() -> pd.DataFrame:
    p = pd.read_csv(os.path.join(DATA, "purchases.csv"), dtype=str)
    p["filed"] = pd.to_datetime(p["FILING_DATE"], format="mixed", errors="coerce")
    p["traded"] = pd.to_datetime(p["TRANS_DATE"], format="mixed", errors="coerce")
    p["shares"] = pd.to_numeric(p["TRANS_SHARES"], errors="coerce")
    p["price"] = pd.to_numeric(p["TRANS_PRICEPERSHARE"], errors="coerce")
    p["value"] = p["shares"] * p["price"]
    rel = p["RPTOWNER_RELATIONSHIP"].fillna("").str.lower()
    p = p[rel.str.contains("officer|director") & (p["price"] > 0)].dropna(subset=["filed", "traded", "value"])
    p["ticker"] = p["ISSUERTRADINGSYMBOL"].fillna("").str.strip().str.upper().str.replace(".", "-", regex=False)
    p = p[p["ticker"].str.fullmatch(r"[A-Z][A-Z\-]{0,5}")]
    return p.drop_duplicates(["ACCESSION_NUMBER", "RPTOWNERCIK", "TRANS_DATE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])


def signals() -> pd.DataFrame:
    p = purchases()
    title = p["RPTOWNER_TITLE"].fillna("").str.lower()
    p["ceo_cfo"] = title.str.contains(r"\bceo\b|chief executive|\bcfo\b|chief financial|president")
    # routine insiders: bought in the same calendar month in each of the 3 previous years (Cohen-Malloy-Pomorski)
    ym = set(zip(p["RPTOWNERCIK"], p["traded"].dt.year, p["traded"].dt.month))
    p["routine"] = [all((o, y - k, m) in ym for k in (1, 2, 3))
                    for o, y, m in zip(p["RPTOWNERCIK"], p["traded"].dt.year, p["traded"].dt.month)]
    f = p.groupby(["ACCESSION_NUMBER", "filed", "ISSUERCIK", "ticker", "RPTOWNERCIK"], as_index=False).agg(
        value=("value", "sum"), shares=("shares", "sum"), px=("price", "median"), traded=("traded", "max"),
        ceo_cfo=("ceo_cfo", "max"), routine=("routine", "min"))
    f = f[f["value"] >= MIN_VALUE].sort_values("filed").reset_index(drop=True)
    rows, last = [], {}
    by_co = {k: g for k, g in f.groupby("ISSUERCIK")}
    for r in f.itertuples():
        if r.ISSUERCIK in last and (r.filed - last[r.ISSUERCIK]).days < 30:
            continue
        last[r.ISSUERCIK] = r.filed
        g = by_co[r.ISSUERCIK]
        recent = g[(g.filed <= r.filed) & (g.filed > r.filed - pd.Timedelta(days=30))]
        same = recent[recent.filed == r.filed]
        rows.append({"cik": r.ISSUERCIK, "ticker": r.ticker, "file_date": r.filed, "trade_date": r.traded,
                     "px": r.px, "value": same.value.sum(), "insiders_30d": recent.RPTOWNERCIK.nunique(),
                     "ceo_cfo": bool(same.ceo_cfo.any()), "opportunistic": not bool(same.routine.all())})
    return pd.DataFrame(rows)


def prices():
    import yfinance as yf
    ev = signals()
    ev.to_csv(os.path.join(DATA, "signals.csv"), index=False)
    tickers = sorted(set(ev["ticker"]) | {"IWM", "SPY"})
    path = os.path.join(DATA, "prices.pkl")
    have = pd.read_pickle(path) if os.path.exists(path) else {"close": pd.DataFrame(), "raw": pd.DataFrame(),
                                                               "volume": pd.DataFrame(), "splits": pd.DataFrame(),
                                                               "tried": set()}
    todo = [t for t in tickers if t not in have["tried"]]
    print(f"{len(ev)} signals, {len(tickers)} tickers, {len(todo)} to download", flush=True)
    import yfinance.shared as yfs
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        try:
            d = yf.download(batch, start="2009-06-01", progress=False, auto_adjust=False, actions=True, threads=4)
        except Exception as e:                       # noqa: BLE001 - a failed batch is retried on the next run
            print(f"  batch failed: {e}")
            time.sleep(60)
            continue
        limited = {t for t, e in yfs._ERRORS.items() if "Rate" in str(e) or "Too Many" in str(e)}
        if limited:                                  # Yahoo is throttling: keep those tickers for the next run
            print(f"  rate-limited on {len(limited)} tickers; pausing", flush=True)
            time.sleep(120)
        close, raw, vol, spl = {}, {}, {}, {}
        for t in batch:
            try:
                c = d["Adj Close"][t].dropna()
            except KeyError:
                continue
            if len(c):
                close[t], raw[t], vol[t] = c, d["Close"][t].reindex(c.index), d["Volume"][t].reindex(c.index)
                spl[t] = d["Stock Splits"][t].reindex(c.index).fillna(0.0)
        for k, v in (("close", close), ("raw", raw), ("volume", vol), ("splits", spl)):
            if v:
                have[k] = pd.concat([have[k], pd.DataFrame(v)], axis=1)
        have["tried"] |= set(batch) - limited
        pd.to_pickle(have, path)
        time.sleep(3)
        print(f"  {i + len(batch)}/{len(todo)}: {have['close'].shape[1]} with prices", flush=True)


def _t(x: pd.Series) -> str:
    x = x.dropna()
    return f"{100 * x.mean():+5.1f}% (t {x.mean() / x.std() * np.sqrt(len(x)):+.1f}, n {len(x)})" if len(x) > 2 else "n/a"


def evaluate():
    ev = pd.read_csv(os.path.join(DATA, "signals.csv"), parse_dates=["file_date", "trade_date"])
    p = pd.read_pickle(os.path.join(DATA, "prices.pkl"))
    close, splits = p["close"], p["splits"]
    bench = close["IWM"].dropna()
    cal = bench.index
    rows = []
    for r in ev.itertuples():
        out = {"priced": False}
        if r.ticker in close.columns and cal[0] < r.file_date < cal[-1]:
            c = close[r.ticker].reindex(cal)
            i0 = cal.searchsorted(r.file_date)
            it = cal.searchsorted(r.trade_date)
            # Yahoo's Close is split-adjusted (Adj Close also for dividends); undo splits after the trade date
            s = splits[r.ticker].reindex(cal).fillna(0.0)
            factor = s[s > 0].loc[cal[min(it, len(cal) - 1)] + pd.Timedelta(days=1):].prod() if (s > 0).any() else 1.0
            raw = p["raw"][r.ticker].reindex(cal).iloc[min(it, len(cal) - 1)] * factor
            if np.isfinite(raw) and 0.8 <= raw / r.px <= 1.25 and i0 + 1 < len(cal) and np.isfinite(c.iloc[i0 + 1]):
                last = c.iloc[i0 + 1:i0 + 2 + HOLD].last_valid_index()
                j = min(i0 + 1 + HOLD, cal.get_loc(last), len(cal) - 1)
                full = i0 + 1 + HOLD < len(cal)
                out = {"priced": True, "complete": full,
                       "ret": c.iloc[j] / c.iloc[i0 + 1] - 1,
                       "abn": c.iloc[j] / c.iloc[i0 + 1] - bench.iloc[j] / bench.iloc[i0 + 1],
                       "control": c.iloc[i0 - 61] / c.iloc[i0 - 121] - bench.iloc[i0 - 61] / bench.iloc[i0 - 121]
                       if i0 >= 121 else np.nan,
                       "dollar_vol": (c * p["volume"][r.ticker].reindex(cal)).iloc[max(i0 - 62, 0):i0 - 2].mean()}
            else:
                out = {"priced": False, "mismatch": np.isfinite(raw)}
        rows.append(out)
    e = pd.concat([ev.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    e["period"] = np.where(e["file_date"] < SPLIT, "2010-18", "2019+")
    e = e[e["file_date"] >= "2010-01-01"]
    print(f"{len(e)} signals 2010+ from {e.cik.nunique()} companies; priced and matched {e.priced.mean():.0%} "
          f"(ticker found but price mismatch {e.get('mismatch', pd.Series(dtype=bool)).fillna(False).astype(bool).mean():.0%})")
    g_all = e[e.priced.fillna(False).astype(bool) & e.complete.fillna(False).astype(bool)].copy()
    g_all["net"] = g_all["abn"] - COST
    small = g_all["dollar_vol"] < 5e6
    groups = {"all": g_all.index == g_all.index, "CEO / CFO": g_all.ceo_cfo, "cluster 3+": g_all.insiders_30d >= 3,
              "opportunistic": g_all.opportunistic, "opportunistic CEO/CFO": g_all.opportunistic & g_all.ceo_cfo,
              "small (< $5M/day traded)": small, "large (>= $5M/day)": ~small}
    print(f"\n60-day return minus IWM, after {COST:.1%} costs (control = same stocks, 60 days ending 3 months earlier)")
    for name, mask in groups.items():
        cells = []
        for per in ("2010-18", "2019+"):
            g = g_all[np.asarray(mask) & (g_all.period == per).to_numpy()]
            cells.append(f"{per} {_t(g.net)} ctrl {_t(g.control)}")
        print(f"  {name:26s} " + " | ".join(cells))
    # calendar-time portfolio: equal weight in every open position, daily, hedged with IWM
    for name in ("all", "opportunistic", "cluster 3+", "opportunistic CEO/CFO"):
        sel = g_all[np.asarray(groups[name])]
        legs = []
        rb = bench.pct_change()
        for r in sel.itertuples():
            i0 = cal.searchsorted(r.file_date)
            seg = slice(i0 + 2, i0 + 2 + HOLD)
            legs.append((close[r.ticker].reindex(cal).pct_change().iloc[seg] - rb.iloc[seg]).clip(-0.9, 3))
        pos = pd.concat(legs, axis=1)
        n = pos.notna().sum(axis=1).reindex(cal).fillna(0)
        port = (pos.sum(axis=1).reindex(cal).fillna(0) / n.replace(0, np.nan)).fillna(0) - (COST / HOLD) * (n > 0)
        cells = []
        for lab, part in (("2010-18", port["2010":"2018"]), ("2019+", port["2019":])):
            eq = (1 + part).cumprod()
            cells.append(f"{lab} {eq.iloc[-1] ** (252 / len(part)) - 1:+6.1%}/yr Sharpe {part.mean() / part.std() * np.sqrt(252):+.2f} "
                         f"DD {(eq / eq.cummax() - 1).min():+.0%}")
        print(f"  portfolio {name:22s} " + " | ".join(cells) + f" | avg positions {n[n > 0].mean():.0f}")


if __name__ == "__main__":
    {"collect": collect, "prices": prices, "evaluate": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "evaluate"]()
