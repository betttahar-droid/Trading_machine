"""
Futures positioning from the CFTC's Commitments of Traders reports (weekly, free, 1986 .. now): the TradFi cousin of
positioning_lab's "smart money vs crowd" finding.

Hedging-pressure theory (Keynes; de Roon, Nijman & Veld 2000; Basu & Miffre 2013): producers hedge by selling
futures, and speculators who take the other side earn a premium. Markets where speculators are most net long (hedgers
most net short) should earn more. Prices: ETFs (they include futures roll costs), 2008 .. now.

  signal     speculators' (non-commercial) net position = (long - short) / (long + short), 52-week average; the report
             (positions as of Tuesday) is used from that Friday's close
  test 1     cross-section: weekly, long the top third of markets and short the bottom third, equal weight, 0.05% per
             unit of turnover; 2008-2016 vs 2017-now; pass = Sharpe > 0.4 in both
  test 2     overlay on the plan's TradFi book (GLD, SLV, SPY, QQQ, 12-month trend): hold a long only when speculators
             are not crowded (their net position's 3-year z-score < 1.5); judged on the plan like the other overlays
  check      the opposite (contrarian) sign is reported too, but only the pre-registered sign counts

    python -m backend.cot_lab
"""

import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cot"))
UA = {"User-Agent": "Mozilla/5.0 (research script; betttahar@gmail.com)"}
# CFTC contract market code -> ETF
MARKETS = {
    "088691": "GLD", "084691": "SLV", "085692": "CPER", "076651": "PPLT", "075651": "PALL",
    "067651": "USO", "023651": "UNG", "111659": "UGA",
    "002602": "CORN", "005602": "SOYB", "001602": "WEAT", "080732": "CANE",
    "13874A": "SPY", "209742": "QQQ", "239742": "IWM", "124603": "DIA",
    "043602": "IEF", "020601": "TLT", "042601": "SHY",
    "099741": "FXE", "097741": "FXY", "096742": "FXB", "090741": "FXC", "232741": "FXA", "092741": "FXF",
}
COST = 0.0005


def cot() -> pd.DataFrame:
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "legacy.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])
    frames = []
    for name in ["deacot1986_2016"] + [f"deacot{y}" for y in range(2017, pd.Timestamp.now().year + 1)]:
        r = requests.get(f"https://www.cftc.gov/files/dea/history/{name}.zip", headers=UA, timeout=120)
        if r.status_code != 200:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        frames.append(pd.DataFrame({
            "code": df["CFTC Contract Market Code"].astype(str).str.strip(),
            "name": df["Market and Exchange Names"],
            "date": pd.to_datetime(df["As of Date in Form YYYY-MM-DD"]),
            "oi": pd.to_numeric(df["Open Interest (All)"], errors="coerce"),
            "spec_long": pd.to_numeric(df["Noncommercial Positions-Long (All)"], errors="coerce"),
            "spec_short": pd.to_numeric(df["Noncommercial Positions-Short (All)"], errors="coerce"),
            "comm_long": pd.to_numeric(df["Commercial Positions-Long (All)"], errors="coerce"),
            "comm_short": pd.to_numeric(df["Commercial Positions-Short (All)"], errors="coerce")}))
        print(f"  {name}: {len(df)} rows", flush=True)
    out = pd.concat(frames)
    out = out[out["code"].isin(MARKETS)].drop_duplicates(["code", "date"]).sort_values("date")
    out.to_csv(path, index=False)
    return out


def weekly_signals():
    c = cot()
    c["net"] = (c.spec_long - c.spec_short) / (c.spec_long + c.spec_short)
    net = c.pivot_table(index="date", columns="code", values="net").rename(columns=MARKETS)
    net.index = net.index + pd.Timedelta(days=3)                 # Tuesday positions, released Friday
    net = net.resample("W-FRI").last()
    return net


def prices() -> pd.DataFrame:
    import yfinance as yf
    path = os.path.join(DATA, "etf_prices.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    px = yf.download(sorted(set(MARKETS.values())), start="2005-01-01", progress=False, auto_adjust=True)["Close"]
    px.to_pickle(path)
    return px


def xs_book(sig: pd.DataFrame, wret: pd.DataFrame, sign: int = 1) -> pd.Series:
    rows, w_prev = [], pd.Series(0.0, index=wret.columns)
    for d in wret.index[:-1]:
        s = (sign * sig.loc[d]).dropna() if d in sig.index else pd.Series(dtype=float)
        s = s[wret.loc[d:].iloc[1:2].notna().iloc[0].reindex(s.index).fillna(False)] if len(s) else s
        w = pd.Series(0.0, index=wret.columns)
        if len(s) >= 6:
            k = len(s) // 3
            r = s.sort_values()
            w[r.index[-k:]] = 0.5 / k
            w[r.index[:k]] = -0.5 / k
        turn = (w - w_prev).abs().sum()
        w_prev = w
        nxt = wret.shift(-1).loc[d].fillna(0.0)
        rows.append((d, (w * nxt).sum() - turn * COST))
    return pd.Series(dict(rows))


def _fmt(r: pd.Series, split: str = "2017-01-01") -> str:
    cells = []
    for lab, x in (("2008-16", r["2008":"2016"]), ("2017+", r[split:])):
        eq = (1 + x).cumprod()
        cells.append(f"{lab} {eq.iloc[-1] ** (52 / len(x)) - 1:+6.1%}/yr Sharpe {x.mean() / x.std() * np.sqrt(52):+.2f} "
                     f"DD {(eq / eq.cummax() - 1).min():+.0%}")
    return " | ".join(cells)


def main():
    net = weekly_signals()
    px = prices()
    wpx = px.resample("W-FRI").last()
    wret = wpx.pct_change(fill_method=None).loc["2007-01-01":]
    net = net.reindex(wret.index)
    print(f"{net.notna().any().sum()} markets with COT data; typical number per week {net.notna().sum(axis=1)['2010':].median():.0f}")
    hp = net.rolling(52, min_periods=40).mean()
    z = (net - net.rolling(156, min_periods=104).mean()) / net.rolling(156, min_periods=104).std()
    print("\nTest 1: cross-section (weekly, top third long / bottom third short)")
    print(f"  {'hedging pressure (pre-registered)':38s} {_fmt(xs_book(hp, wret, +1))}")
    print(f"  {'same, opposite sign (check)':38s} {_fmt(xs_book(hp, wret, -1))}")
    print(f"  {'3-year z-score, contrarian (check)':38s} {_fmt(xs_book(z, wret, -1))}")
    print(f"  {'3-year z-score, follow (check)':38s} {_fmt(xs_book(z, wret, +1))}")
    past = wpx / wpx.shift(52) - 1
    print(f"  {'12-month momentum (reference)':38s} {_fmt(xs_book(past.reindex(wret.index), wret, +1))}")
    zz, pp = z.sub(z.mean(axis=1), axis=0), past.reindex(wret.index)
    print(f"  cross-sectional correlation of hedging pressure with 12-month momentum: "
          f"{hp.rank(axis=1).corrwith(pp.rank(axis=1), axis=1).mean():+.2f}")

    # Test 2: crowding overlay on the plan's TradFi book
    from backend.cross_asset_lab import BINANCE_TRADFI, FUNDING_2026, crypto_trend_returns, etf_prices, tsmom_returns
    from backend.strategy_lab import START, fmt
    daily = etf_prices("2005-01-01")[list(BINANCE_TRADFI)]
    zd = z[list(BINANCE_TRADFI)].reindex(daily.index, method="ffill")
    crowded = zd > 1.5
    print(f"\nTest 2: share of days each plan asset counts as crowded: {crowded['2008':].mean().round(2).to_dict()}")
    base = tsmom_returns(daily, FUNDING_2026)
    # a crowded asset is held flat: its trend weight is zeroed by making its 12-month signal non-positive
    ret = daily.pct_change(fill_method=None)
    held_base = tsmom_weights(daily)
    held = held_base.where(~crowded.shift(1).fillna(False).astype(bool), 0.0)
    over = (held * ret.fillna(0)).sum(axis=1) - (held - held.shift(1).fillna(0)).abs().sum(axis=1) * COST \
        - (held * pd.Series(FUNDING_2026).reindex(held.columns).fillna(0) / 252).sum(axis=1)
    for lab, b in (("TradFi book (current)", base), ("TradFi book, skip crowded longs", over)):
        x = b["2008":]
        print(f"  {lab:34s} 2008-16 Sharpe {x[:'2016'].mean() / x[:'2016'].std() * np.sqrt(252):+.2f} | "
              f"2017+ Sharpe {x['2017':].mean() / x['2017':].std() * np.sqrt(252):+.2f}")
    crypto = crypto_trend_returns()
    days = pd.date_range(crypto.index.min(), crypto.index.max(), freq="D")
    cd = crypto.reindex(days).fillna(0.0)
    for lab, b in (("plan (current)", base), ("plan, TradFi skips crowded longs", over)):
        both = pd.DataFrame({"c": cd, "t": b.reindex(days).fillna(0.0)})
        vol_is = both[START:"2024-06-30"].std()
        w = (1 / vol_is) / (1 / vol_is).sum()
        combo = (both * w).sum(axis=1)
        combo *= cd[START:"2024-06-30"].std() / combo[START:"2024-06-30"].std()
        print(fmt(lab, combo))


def tsmom_weights(px: pd.DataFrame) -> pd.DataFrame:
    """The held weights of cross_asset_lab.tsmom_returns (decided at month end, held from the next day)."""
    from backend.cross_asset_lab import TARGET_VOL
    ret = px.pct_change(fill_method=None)
    mom = px / px.shift(252) - 1
    vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    month_end = px.index.to_series().groupby(px.index.to_period("M")).transform("max") == px.index.to_series()
    n = px.notna().sum(axis=1).clip(lower=1)
    target = ((mom > 0).astype(float) * (TARGET_VOL / np.sqrt(n)).values[:, None] / vol).where(month_end)
    return target.ffill().fillna(0.0).clip(upper=1.0).shift(1).fillna(0.0)


if __name__ == "__main__":
    main()
