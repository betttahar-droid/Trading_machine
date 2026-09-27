"""
Selling Bitcoin volatility: does the volatility risk premium pay, does it balance the trend strategy, and can a
volatility forecast (the one thing the models forecast a little better than naive rules) time it?

Every Friday 08:00 UTC (Deribit's weekly expiry) sell a 7-day structure on BTC, priced with Black-Scholes at
Deribit's DVOL (30-day implied volatility index, known at the prior day's close), held to expiry:
  straddle    sell the at-the-money call and put (unlimited risk)
  iron fly    the same plus bought wings 10% away (maximum loss about 10% of the underlying minus the premium)
Costs: 1 volatility point worse than DVOL on every leg; bought puts a further 5 points and bought calls 2 points
dearer (crypto skew); Deribit fee min(0.03% of underlying, 12.5% of the option price) per leg.
In-sample 2021-04 .. 2024-06, out-of-sample 2024-07 .. now, like the other labs.

Timing: sell only when DVOL / forecast volatility is at or above its in-sample median. Forecasts: the last 30 days'
realised volatility, and Chronos-Bolt's 7-day forecast spread.

    python -m backend.vol_premium_lab
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from backend.chronos_lab import closes
from backend.insight_lab import dvol

T = 7 / 365
WING = 0.10
SPLIT = "2024-07-01"


def bs(S, K, sig, call):
    d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2) if call else K * norm.cdf(-d2) - S * norm.cdf(-d1)


def fee(S, price):
    return min(0.0003 * S, 0.125 * price)


def weekly_trades(btc: pd.Series, iv: pd.Series, term: float = 0.0, half_spread: float = 0.01) -> pd.DataFrame:
    """term: how far 7-day implied volatility sits below DVOL; half_spread: vol points lost per leg."""
    rows = []
    fridays = btc.index[(btc.index.dayofweek == 4) & (btc.index.hour == 8)]
    for t in fridays:
        t1 = t + pd.Timedelta(days=7)
        d = (t - pd.Timedelta(days=1)).normalize()
        if t1 not in btc.index or d not in iv.index:
            continue
        S, ST, sig = btc[t], btc[t1], iv[d] / 100
        sig = sig - term                                      # 7-day level
        sell = sig - half_spread
        c, p = bs(S, S, sell, True), bs(S, S, sell, False)
        straddle_prem = c + p - fee(S, c) - fee(S, p)
        payoff = abs(ST - S)
        wc = bs(S, S * (1 + WING), sig + half_spread + 0.02, True)
        wp = bs(S, S * (1 - WING), sig + half_spread + 0.05, False)
        wing_cost = wc + wp + fee(S, wc) + fee(S, wp)
        wing_payoff = max(ST - S * (1 + WING), 0) + max(S * (1 - WING) - ST, 0)
        past = np.log(btc[:t].iloc[-181:]).diff().dropna()           # last 30 days of 4h returns
        rows.append({"t": t, "iv": sig, "rv_next": np.log(btc[t:t1]).diff().dropna().std() * np.sqrt(6 * 365),
                     "rv_past": past.std() * np.sqrt(6 * 365),
                     "straddle": (straddle_prem - payoff) / S,
                     "iron_fly": (straddle_prem - payoff - wing_cost + wing_payoff) / S})
    return pd.DataFrame(rows).set_index("t")


def chronos_rv(btc: pd.Series, when: pd.DatetimeIndex, model="amazon/chronos-bolt-small") -> pd.Series:
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(model, device_map="cpu", torch_dtype=torch.float32)
    out = {}
    for t in when:
        hist = btc[:t].iloc[-512:]
        q, _ = pipe.predict_quantiles([torch.tensor(hist.to_numpy(), dtype=torch.float32)], prediction_length=42,
                                      quantile_levels=[0.1, 0.9])
        spread = float(q[0, -1, 1] - q[0, -1, 0]) / hist.iloc[-1]
        out[t] = spread / (2 * 1.2816) * np.sqrt(365 / 7)             # 10-90% range of the 7-day move -> yearly vol
    return pd.Series(out)


def show(name, r: pd.Series):
    def one(x):
        if len(x) < 10:
            return "n/a"
        eq = (1 + x).cumprod()
        return (f"{(eq.iloc[-1] ** (52 / len(x)) - 1):+7.1%}/yr  Sharpe {x.mean() / x.std() * np.sqrt(52):+.2f}  "
                f"worst week {x.min():+.1%}  max DD {(eq / eq.cummax() - 1).min():+.0%}")
    print(f"  {name:44s} in {one(r[:SPLIT])} | out {one(r[SPLIT:])}")


def main():
    from backend.cross_asset_lab import BINANCE_TRADFI, FUNDING_2026, crypto_trend_returns, etf_prices, tsmom_returns
    from backend.strategy_lab import START
    btc = closes()["BTCUSDT"].dropna()
    btc.index = btc.index + pd.Timedelta(hours=4)                    # 4h bar close time
    w = weekly_trades(btc, dvol())
    print(f"{len(w)} weeks {w.index[0]:%Y-%m-%d} .. {w.index[-1]:%Y-%m-%d}")
    for lab, part in (("in-sample", w[:SPLIT]), ("out-of-sample", w[SPLIT:])):
        print(f"  {lab}: implied {part.iv.mean():.1%} vs realised next week {part.rv_next.mean():.1%} "
              f"(implied higher in {(part.iv > part.rv_next).mean():.0%} of weeks)")

    # the plan (crypto trend + TradFi) as weekly returns, Friday 08:00 to Friday 08:00 (daily data -> Friday close)
    crypto = crypto_trend_returns()
    days = pd.date_range(crypto.index.min(), crypto.index.max(), freq="D")
    cd = crypto.reindex(days).fillna(0.0)
    tb = tsmom_returns(etf_prices("2005-01-01")[list(BINANCE_TRADFI)], FUNDING_2026).reindex(days).fillna(0.0)
    both = pd.DataFrame({"c": cd, "t": tb})
    vol_is = both[START:"2024-06-30"].std()
    wts = (1 / vol_is) / (1 / vol_is).sum()
    plan = (both * wts).sum(axis=1)
    plan *= cd[START:"2024-06-30"].std() / plan[START:"2024-06-30"].std()
    plan_w = (1 + plan).groupby(pd.Grouper(freq="W-FRI")).prod() - 1
    plan_w.index = plan_w.index + pd.Timedelta(hours=8)

    print("\nWeekly returns; option books sized to the plan's in-sample volatility (plan = level 1)")
    rv_c = chronos_rv(btc, w.index)
    for struct in ("straddle", "iron_fly"):
        x = w[struct]
        L = plan_w[:SPLIT].std() / x[:SPLIT].std()
        book = L * x
        print(f"\n  {struct} (notional {L:.2f}x capital)")
        show("always sell", book)
        for fname, fc in (("last 30 days' volatility", w["rv_past"]), ("Chronos 7-day forecast", rv_c)):
            ratio = w["iv"] / fc.reindex(w.index)
            thr = ratio[:SPLIT].median()
            timed = book.where(ratio >= thr, 0.0)
            show(f"sell when implied/forecast >= {thr:.2f} ({fname})", timed)
        j = pd.concat({"plan": plan_w, "opt": book}, axis=1).dropna()
        print(f"  correlation with the plan: in {j[:SPLIT].corr().iloc[0, 1]:+.2f} | out {j[SPLIT:].corr().iloc[0, 1]:+.2f}")
        show("plan alone", j["plan"])
        show("plan + option book, half each", 0.5 * j["plan"] + 0.5 * j["opt"])
        show("plan + option book, 100% + 50%", j["plan"] + 0.5 * j["opt"])


if __name__ == "__main__":
    main()
