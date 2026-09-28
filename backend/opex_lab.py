"""
Monthly options expiration (OPEX) in US stocks. Traders' folklore and some papers (e.g. Stivers & Sun 2013) say the
week of the third-Friday expiration is strong (dealers' hedging pins and supports prices) and the week after is weak.
Fixed in advance: weekly SPY (1993+) and QQQ (1999+) returns, Friday close to Friday close; OPEX week = the week ending
on the month's third Friday (or the day before when it is a holiday). Counts only if the OPEX-week minus other-week
difference has t >= 2 in both 1993-2009 and 2010-now with the same sign.

    python -m backend.opex_lab
"""

import numpy as np
import pandas as pd


def main():
    import yfinance as yf
    px = yf.download(["SPY", "QQQ"], start="1993-01-01", progress=False, auto_adjust=True)["Close"]
    for t in ("SPY", "QQQ"):
        c = px[t].dropna()
        wk = c.resample("W-FRI").last()
        r = wk.pct_change().dropna()
        third_fri = pd.Series([d for d in pd.date_range(c.index[0], c.index[-1], freq="WOM-3FRI")])
        opex = r.index.isin(third_fri)
        after = r.index.isin(third_fri + pd.Timedelta(days=7))
        for lab, lo, hi in (("1993-2009", "1993", "2009"), ("2010-now ", "2010", "2100")):
            x, o, a = r[lo:hi], opex[(r.index >= lo) & (r.index <= f"{hi}-12-31")], after[(r.index >= lo) & (r.index <= f"{hi}-12-31")]
            other = x[~o & ~a]

            def diff(m):
                d = x[m].mean() - other.mean()
                se = np.sqrt(x[m].var() / m.sum() + other.var() / len(other))
                return f"{100 * x[m].mean():+.2f}% (vs others {100 * d:+.2f}%, t {d / se:+.1f}, n {m.sum()})"
            print(f"{t} {lab} OPEX week {diff(o)} | week after {diff(a)} | other weeks {100 * other.mean():+.2f}%")


if __name__ == "__main__":
    main()
