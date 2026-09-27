"""
Insider buying in drug / biotech companies: executives and directors know how their trials are going, and open-market
purchases by insiders are one of the best-documented stock signals (Lakonishok & Lee 2001; Jeng, Metrick & Zeckhauser
2003; Cohen, Malloy & Pomorski 2012), strongest in small companies.

Data: the SEC's quarterly Insider Transactions Data Sets (Forms 3/4/5, 2015 .. now), restricted to the ~900 drug /
biotech companies found by the other biotech labs; open-market purchases (transaction code P) by officers or
directors. Prices: Yahoo; abnormal return vs XBI.

Rules fixed before looking at results:
  signal  a Form 4 filing with officer / director open-market purchases worth >= $10,000 (one signal per company per
          30 days); entry at the close of the first trading day after the filing date
  trade   hold 60 trading days, long the stock and short XBI, 0.5% round-trip costs
  pass    positive net in 2015-21 and in 2022+; control = the same stocks over 60 days ending 3 months earlier
  second  "cluster" buys (2+ different insiders within 30 days), and 120-day holding

    python -m backend.biotech_insider_lab collect
    python -m backend.biotech_insider_lab evaluate
"""

import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from backend import biotech_lab as bl

DATA = os.path.join(os.path.dirname(bl.DATA), "biotech_insider")
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"


def biotech_ciks() -> set:
    ciks = set()
    for lab in ("biotech", "biotech_start", "biotech_pdufa"):
        path = os.path.join(os.path.dirname(bl.DATA), lab, "hits.csv")
        if os.path.exists(path):
            ciks |= set(pd.read_csv(path, usecols=["cik"])["cik"].astype(int))
    return ciks


def collect():
    os.makedirs(DATA, exist_ok=True)
    ciks = biotech_ciks()
    print(f"{len(ciks)} drug / biotech companies")
    out = []
    quarters = [f"{y}q{q}" for y in range(2015, int(time.strftime("%Y")) + 1) for q in range(1, 5)]
    for q in quarters:
        path = os.path.join(DATA, f"{q}.csv")
        if os.path.exists(path):
            out.append(pd.read_csv(path))
            continue
        r = bl._get(URL.format(q=q))
        if r is None:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = {n.split("/")[-1].upper(): n for n in z.namelist()}

        def tab(name, cols):
            return pd.read_csv(z.open(names[name]), sep="\t", usecols=cols, dtype=str, on_bad_lines="skip")
        sub = tab("SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "DOCUMENT_TYPE"])
        sub = sub[sub["ISSUERCIK"].astype(int).isin(ciks) & sub["DOCUMENT_TYPE"].isin(["4", "4/A"])]
        tr = tab("NONDERIV_TRANS.TSV", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                         "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
        tr = tr[(tr["TRANS_CODE"] == "P") & (tr["TRANS_ACQUIRED_DISP_CD"] == "A")]
        own = tab("REPORTINGOWNER.TSV", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
        m = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER")
        m["value"] = pd.to_numeric(m["TRANS_SHARES"], errors="coerce") * pd.to_numeric(m["TRANS_PRICEPERSHARE"], errors="coerce")
        m = m[["ACCESSION_NUMBER", "FILING_DATE", "TRANS_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "RPTOWNERCIK",
               "RPTOWNER_RELATIONSHIP", "value"]]
        m.to_csv(path, index=False)
        out.append(m)
        print(f"  {q}: {len(m)} purchase rows", flush=True)
        time.sleep(0.5)
    pd.concat(out).to_csv(os.path.join(DATA, "purchases.csv"), index=False)


def signals() -> pd.DataFrame:
    p = pd.read_csv(os.path.join(DATA, "purchases.csv"), dtype={"ISSUERTRADINGSYMBOL": str})
    rel = p["RPTOWNER_RELATIONSHIP"].fillna("").str.lower()
    p = p[rel.str.contains("officer|director")]
    p["filed"] = pd.to_datetime(p["FILING_DATE"], format="mixed", errors="coerce")
    p = p.dropna(subset=["filed"])
    per_filing = p.groupby(["ACCESSION_NUMBER", "filed", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "RPTOWNERCIK"],
                           as_index=False)["value"].sum()
    per_filing = per_filing[per_filing["value"] >= 10_000].sort_values("filed")
    rows, last = [], {}
    for r in per_filing.itertuples():
        # insiders buying the same company within the past 30 days (cluster)
        recent = per_filing[(per_filing.ISSUERCIK == r.ISSUERCIK) & (per_filing.filed <= r.filed)
                            & (per_filing.filed > r.filed - pd.Timedelta(days=30))]
        if r.ISSUERCIK in last and (r.filed - last[r.ISSUERCIK]).days < 30:
            continue
        last[r.ISSUERCIK] = r.filed
        rows.append({"cik": r.ISSUERCIK, "ticker": str(r.ISSUERTRADINGSYMBOL).strip().upper(), "file_date": r.filed,
                     "value": r.value, "insiders_30d": recent.RPTOWNERCIK.nunique()})
    return pd.DataFrame(rows)


def evaluate():
    ev = signals()
    from backend.biotech_start_lab import update_prices
    update_prices(ev)
    p = pd.read_pickle(os.path.join(bl.DATA, "prices.pkl"))
    close, xbi = p["close"], p["close"]["XBI"].dropna()
    cal, rx = xbi.index, xbi.pct_change()
    rows = []
    for r in ev.itertuples():
        if r.ticker not in close.columns:
            rows.append({"has_price": False})
            continue
        c = close[r.ticker].reindex(cal)
        i0 = cal.searchsorted(r.file_date)

        def ar(a, b):
            if a < 0 or b >= len(cal):
                return np.nan
            x = c.iloc[b] / c.iloc[a] - 1
            return x - (xbi.iloc[b] / xbi.iloc[a] - 1) if np.isfinite(x) else np.nan
        last = c.iloc[i0 + 1:i0 + 122].last_valid_index()                   # delisted inside the window
        j = (lambda h: min(i0 + 1 + h, cal.get_loc(last)) if last is not None else i0 + 1)
        rows.append({"has_price": np.isfinite(c.iloc[i0 + 1]) if i0 + 1 < len(cal) else False,
                     "post60": ar(i0 + 1, j(60)), "post120": ar(i0 + 1, j(120)), "control": ar(i0 - 121, i0 - 61),
                     "dollar_vol": (c * p["volume"][r.ticker].reindex(cal)).iloc[max(i0 - 62, 0):i0 - 2].mean()})
    ev = pd.concat([ev.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    ev["period"] = np.where(ev["file_date"] < "2022-01-01", "2015-21", "2022+")
    e = ev[ev.has_price.fillna(False).astype(bool)].copy()
    print(f"{len(ev)} insider-buying signals from {ev.cik.nunique()} companies; {len(e) / len(ev):.0%} with prices")
    small = e["dollar_vol"] < e.loc[e.period == "2015-21", "dollar_vol"].median()
    for per in ("2015-21", "2022+"):
        g = e[e.period == per]
        print(f"\n  {per}: next 60d {bl._t(g.post60)} | net {bl._t(g.post60 - 0.005)} | median {100 * g.post60.median():+.1f}%")
        print(f"          control {bl._t(g.control)} | next 120d {bl._t(g.post120)}")
        print(f"          cluster (2+ insiders) 60d {bl._t(g[g.insiders_30d >= 2].post60)} | single {bl._t(g[g.insiders_30d < 2].post60)}")
        print(f"          small firms 60d {bl._t(g[small[g.index]].post60)} | large {bl._t(g[~small[g.index]].post60)}")
    legs = []
    for r in e.itertuples():
        i0 = cal.searchsorted(r.file_date)
        seg = slice(i0 + 2, i0 + 62)
        legs.append((close[r.ticker].reindex(cal).pct_change().iloc[seg] - rx.iloc[seg]).fillna(0.0))
    pos = pd.concat(legs, axis=1)
    n = pos.notna().sum(axis=1).reindex(cal).fillna(0)
    port = (pos.sum(axis=1).reindex(cal).fillna(0) / n.replace(0, np.nan)).fillna(0) - (0.005 / 60) * (n > 0)
    for lab, part in (("2015-21", port["2015":"2021"]), ("2022+", port["2022":])):
        eq = (1 + part).cumprod()
        print(f"  portfolio {lab}: {(eq.iloc[-1] ** (252 / len(part)) - 1):+.1%}/yr, Sharpe "
              f"{part.mean() / part.std() * np.sqrt(252):+.2f}, max DD {(eq / eq.cummax() - 1).min():+.0%}, "
              f"avg positions {n.reindex(part.index)[n.reindex(part.index) > 0].mean():.1f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if cmd == "collect":
        collect()
    else:
        evaluate()
