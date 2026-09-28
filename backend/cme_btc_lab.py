"""
Wall Street positioning in CME Bitcoin futures (CFTC Traders in Financial Futures report, weekly, 2018 .. now) as a
timing signal for Bitcoin. Asset managers (funds, ETF-like money) vs leveraged funds (hedge funds, mostly running the
cash-and-carry basis trade). Fixed in advance:

  signal    4-week change in asset managers' net long as a share of open interest (institutional demand building);
            the report (positions as of Tuesday) is used from that Friday's close
  rule      hold Bitcoin the next week only when the signal is positive; compared with always holding it
  periods   2018-2022 vs 2023-now; pass = better Sharpe than holding in both
  check     the same with leveraged funds' net position (4-week change), both signs shown

    python -m backend.cme_btc_lab
"""

import io
import os
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from backend.universe_data import load_bars

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cot"))
UA = {"User-Agent": "Mozilla/5.0 (research script; betttahar@gmail.com)"}
BTC_CODE = "133741"


def tff() -> pd.DataFrame:
    path = os.path.join(DATA, "tff_btc.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    os.makedirs(DATA, exist_ok=True)
    frames = []
    for y in range(2017, pd.Timestamp.now().year + 1):
        r = requests.get(f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{y}.zip", headers=UA, timeout=120)
        if r.status_code != 200:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        df = df[df["CFTC_Contract_Market_Code"].astype(str).str.strip() == BTC_CODE]
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(df["Report_Date_as_YYYY-MM-DD"]),
            "oi": pd.to_numeric(df["Open_Interest_All"], errors="coerce"),
            "am": pd.to_numeric(df["Asset_Mgr_Positions_Long_All"], errors="coerce")
            - pd.to_numeric(df["Asset_Mgr_Positions_Short_All"], errors="coerce"),
            "lev": pd.to_numeric(df["Lev_Money_Positions_Long_All"], errors="coerce")
            - pd.to_numeric(df["Lev_Money_Positions_Short_All"], errors="coerce")}))
    out = pd.concat(frames).drop_duplicates("date").sort_values("date")
    out.to_csv(path, index=False)
    return out


def main():
    t = tff().set_index("date")
    t.index = t.index + pd.Timedelta(days=3)                   # released Friday
    w = t.resample("W-FRI").last()
    am = (w["am"] / w["oi"]).diff(4)
    lev = (w["lev"] / w["oi"]).diff(4)
    btc = load_bars("BTCUSDT", "1d", time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400)))["close"]
    wr = btc.resample("W-FRI").last().pct_change()
    nxt = wr.shift(-1).reindex(w.index)
    print(f"{w['oi'].notna().sum()} weekly reports, {w.index.min():%Y-%m} .. {w.index.max():%Y-%m}")

    def show(lab, s):
        cells = []
        for per, lo, hi in (("2018-22", "2018", "2022"), ("2023+", "2023", "2100")):
            x, sg = nxt[lo:hi].dropna(), s[lo:hi]
            held = x.where(sg.reindex(x.index) > 0, 0.0)
            f = lambda r: r.mean() / r.std() * np.sqrt(52)
            cells.append(f"{per} hold-when-positive Sharpe {f(held):+.2f} vs always {f(x):+.2f} "
                         f"(in the market {(sg.reindex(x.index) > 0).mean():.0%})")
        print(f"  {lab:34s} " + " | ".join(cells))
    show("asset managers adding (pre-reg.)", am)
    show("asset managers cutting (check)", -am)
    show("leveraged funds adding (check)", lev)
    show("leveraged funds cutting (check)", -lev)


if __name__ == "__main__":
    main()
