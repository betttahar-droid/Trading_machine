"""
Can a small local language model read financial statements and predict whether a company's earnings rise next year?
(Kim, Muhn & Nikolaev 2024, "Financial Statement Analysis with Large Language Models": GPT-4 on anonymised
statements predicted the direction of next year's earnings about as well as a trained neural net, and portfolios
sorted on its predictions earned alpha.) Here: models that fit a 4 GB RTX 3050.

Data: SEC Financial Statement Data Sets (free; every 10-K's XBRL numbers). Q1 + Q2 files of 2016 .. now cover most
annual reports. One row per 10-K: this year's and last year's key items; target = next 10-K's EPS > this EPS.

Stages (stop at the first that fails):
  1 baseline  logistic regression on standard ratios, trained on fiscal years 2016-2021, tested on 2022-2024
  2 model     zero-shot letter log-probabilities on the anonymised two-year statements (fin_statement_score.py,
              rented GPU); pass = test AUC above the baseline's, or a significant gain when combined with it
  3 returns   only if 2 passes

    python -m backend.fin_statement_lab collect
    python -m backend.fin_statement_lab baseline
"""

import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from backend import biotech_lab as bl

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "fsds"))
URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"
ITEMS = {
    "Revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "CostOfRevenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "GrossProfit": ["GrossProfit"],
    "OperatingIncome": ["OperatingIncomeLoss"],
    "InterestExpense": ["InterestExpense"],
    "IncomeTax": ["IncomeTaxExpenseBenefit"],
    "NetIncome": ["NetIncomeLoss", "ProfitLoss"],
    "EPS": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "DilutedShares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "Cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "CurrentAssets": ["AssetsCurrent"],
    "TotalAssets": ["Assets"],
    "CurrentLiabilities": ["LiabilitiesCurrent"],
    "LongTermDebt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "TotalLiabilities": ["Liabilities"],
    "Equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
    "Capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "Depreciation": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation"],
    "Dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
}
TAG2ITEM = {t: (item, rank) for item, tags in ITEMS.items() for rank, t in enumerate(tags)}
FLOW = {"Revenue", "CostOfRevenue", "GrossProfit", "OperatingIncome", "InterestExpense", "IncomeTax", "NetIncome", "EPS",
        "DilutedShares", "OperatingCashFlow", "Capex", "Depreciation", "Dividends"}


def _quarter(q: str) -> pd.DataFrame:
    r = bl._get(URL.format(q=q))
    if r is None:
        return pd.DataFrame()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    sub = pd.read_csv(z.open("sub.txt"), sep="\t", dtype=str, usecols=["adsh", "cik", "sic", "form", "period", "fy", "fp", "filed"])
    sub = sub[(sub.form == "10-K") & (sub.fp == "FY")]
    keep = set(sub.adsh)
    rows = []
    for chunk in pd.read_csv(z.open("num.txt"), sep="\t", dtype=str, chunksize=2_000_000,
                             usecols=["adsh", "tag", "ddate", "qtrs", "uom", "coreg", "value"], on_bad_lines="skip"):
        c = chunk[chunk.adsh.isin(keep) & chunk.tag.isin(TAG2ITEM) & chunk.coreg.isna()]
        rows.append(c)
    num = pd.concat(rows)
    num["item"] = num.tag.map(lambda t: TAG2ITEM[t][0])
    num["rank"] = num.tag.map(lambda t: TAG2ITEM[t][1])
    num["value"] = pd.to_numeric(num.value, errors="coerce")
    flow = num.item.isin(FLOW)
    num = num[(flow & (num.qtrs == "4")) | (~flow & (num.qtrs == "0"))]
    num = num.merge(sub[["adsh", "period"]], on="adsh")
    per = pd.to_datetime(num.period, format="%Y%m%d")
    dd = pd.to_datetime(num.ddate, format="%Y%m%d")
    num["col"] = np.where(dd == per, "t", np.where((per - dd).dt.days.between(350, 380), "t1", None))
    num = num.dropna(subset=["col"]).sort_values("rank").drop_duplicates(["adsh", "item", "col"])
    wide = num.pivot_table(index="adsh", columns=["col", "item"], values="value", aggfunc="first")
    wide.columns = [f"{i}_{c}" for c, i in wide.columns]
    return sub.set_index("adsh").join(wide, how="inner").reset_index()


def collect():
    os.makedirs(DATA, exist_ok=True)
    parts = []
    for y in range(2016, int(time.strftime("%Y")) + 1):
        for qn in (1, 2):
            q = f"{y}q{qn}"
            path = os.path.join(DATA, f"{q}.pkl")
            if not os.path.exists(path):
                df = _quarter(q)
                if df.empty:
                    continue
                df.to_pickle(path)
                print(f"  {q}: {len(df)} 10-Ks", flush=True)
            parts.append(pd.read_pickle(path))
    df = pd.concat(parts, ignore_index=True).drop_duplicates("adsh")
    df["fy"] = pd.to_numeric(df.fy, errors="coerce")
    df = df.sort_values("filed").drop_duplicates(["cik", "fy"], keep="first")
    nxt = df[["cik", "fy", "EPS_t"]].rename(columns={"EPS_t": "EPS_next"})
    nxt["fy"] -= 1
    df = df.merge(nxt, on=["cik", "fy"], how="left")
    df.to_pickle(os.path.join(DATA, "tenk.pkl"))
    print(f"{len(df)} 10-Ks, {df.EPS_next.notna().sum()} with next year's EPS")


def features(df: pd.DataFrame) -> pd.DataFrame:
    a = df["TotalAssets_t"].where(df["TotalAssets_t"] > 0)
    f = pd.DataFrame(index=df.index)
    f["eps_up_now"] = (df.EPS_t > df.EPS_t1).astype(float)
    f["deps"] = ((df.EPS_t - df.EPS_t1) / df.EPS_t1.abs().clip(lower=0.1)).clip(-3, 3)
    f["loss"] = (df.NetIncome_t < 0).astype(float)
    f["roa"] = (df.NetIncome_t / a).clip(-2, 2)
    f["accruals"] = ((df.NetIncome_t - df.OperatingCashFlow_t) / a).clip(-2, 2)
    f["sales_growth"] = (df.Revenue_t / df.Revenue_t1 - 1).clip(-1, 3)
    f["gross_margin_chg"] = (df.GrossProfit_t / df.Revenue_t - df.GrossProfit_t1 / df.Revenue_t1).clip(-1, 1)
    f["leverage"] = (df.TotalLiabilities_t / a).clip(0, 3)
    f["current_ratio"] = (df.CurrentAssets_t / df.CurrentLiabilities_t).clip(0, 10)
    f["capex"] = (df.Capex_t / a).clip(0, 1)
    f["dividend"] = (df.Dividends_t.fillna(0) > 0).astype(float)
    f["size"] = np.log(a)
    f["eps_level"] = df.EPS_t.clip(-20, 20)
    return f


def dataset() -> pd.DataFrame:
    df = pd.read_pickle(os.path.join(DATA, "tenk.pkl"))
    df = df[df.EPS_t.notna() & df.EPS_t1.notna() & df.EPS_next.notna() & df.TotalAssets_t.notna() & df.NetIncome_t.notna()]
    df = df[df.EPS_next != df.EPS_t].copy()
    df["y"] = (df.EPS_next > df.EPS_t).astype(int)
    return df.reset_index(drop=True)


def baseline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    df = dataset()
    X = features(df).fillna(0.0)
    tr, te = df.fy.between(2016, 2021), df.fy.between(2022, 2024)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], df.y[tr])
    p = clf.predict_proba(X)[:, 1]
    df["p_base"] = p
    df.to_pickle(os.path.join(DATA, "dataset.pkl"))
    print(f"{len(df)} firm-years with an outcome; train FY2016-21 {tr.sum()}, test FY2022-24 {te.sum()}")
    print(f"  base rate (EPS up) train {df.y[tr].mean():.1%} | test {df.y[te].mean():.1%}")
    print(f"  logistic baseline: test AUC {roc_auc_score(df.y[te], p[te]):.3f}, accuracy {((p[te] > 0.5) == df.y[te]).mean():.1%}")
    naive = 1 - X["eps_up_now"]
    print(f"  naive 'earnings mean-revert' rule: test AUC {roc_auc_score(df.y[te], naive[te]):.3f}, "
          f"accuracy {((naive[te] > 0.5) == df.y[te]).mean():.1%}")


ROWS = [("Revenue", "Revenue"), ("CostOfRevenue", "Cost of revenue"), ("GrossProfit", "Gross profit"),
        ("OperatingIncome", "Operating income"), ("InterestExpense", "Interest expense"), ("IncomeTax", "Income tax"),
        ("NetIncome", "Net income"), ("EPS", "Diluted EPS (USD)"), ("DilutedShares", "Diluted shares (millions)"),
        ("Cash", "Cash and equivalents"), ("CurrentAssets", "Current assets"), ("TotalAssets", "Total assets"),
        ("CurrentLiabilities", "Current liabilities"), ("LongTermDebt", "Long-term debt"),
        ("TotalLiabilities", "Total liabilities"), ("Equity", "Shareholders' equity"),
        ("OperatingCashFlow", "Operating cash flow"), ("Capex", "Capital expenditure"),
        ("Depreciation", "Depreciation and amortisation"), ("Dividends", "Dividends paid")]


def table(r) -> str:
    lines = ["Item | Previous year | Current year"]
    for key, label in ROWS:
        a, b = r.get(f"{key}_t1"), r.get(f"{key}_t")
        if pd.isna(a) and pd.isna(b):
            continue
        if key == "EPS":
            fmt = lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"
        else:
            fmt = lambda v: "n/a" if pd.isna(v) else f"{v / 1e6:,.1f}"
        lines.append(f"{label} | {fmt(a)} | {fmt(b)}")
    return "\n".join(lines)


def write_input():
    df = pd.read_pickle(os.path.join(DATA, "dataset.pkl"))
    df = df[df.fy.between(2019, 2024)]
    out = os.path.join(DATA, "input.jsonl")
    df.assign(id=df.adsh, table=[table(r) for r in df.to_dict("records")])[["id", "table"]].to_json(
        out, orient="records", lines=True)
    print(f"{len(df)} statements -> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    {"collect": collect, "input": write_input, "baseline": baseline}.get(cmd, baseline)()
