"""
Stocks + crypto: does adding a stock/ETF trend book raise the Sharpe ratio of the crypto trend strategy?

Growth is limited by the Sharpe ratio (see CLAUDE_HANDOVER.md), and the textbook way to raise it is to add a
strategy that earns on its own but moves independently. Time-series momentum (Moskowitz, Ooi & Pedersen 2012)
across asset classes is the best-documented candidate:

  universe   12 liquid US ETFs (daily, Yahoo Finance): SPY QQQ IWM EFA EEM (stocks), TLT IEF (bonds),
             GLD SLV (metals), DBC (commodities), VNQ (real estate), UUP (US dollar)
  rule       at each month end, hold an ETF only if its past 12-month return is positive (long / flat), sized to
             an equal share of a 12% yearly volatility target using its 60-day volatility; costs 0.05% per trade
  combined   equal-risk mix with the live crypto trend strategy (weights fixed in advance, nothing fitted)

Reported for 2020-06 .. 2024-06 and 2024-07 .. now like the other labs, plus the stock book's own long history.

    python -m backend.cross_asset_lab
"""

import time

import numpy as np
import pandas as pd

from backend.backtest_trend import UNIVERSE, _ms, simulate
from backend.growth_study import LIVE, load_market_bulk
from backend.levers_lab import bootstrap_odds
from backend.strategy_lab import START, fmt, stats
from backend.universe_data import available_months

ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "DBC", "VNQ", "UUP"]
# What Binance Futures lists as USDT "TradFi" perpetuals (since 2026): S&P 500, Nasdaq 100, gold, silver
BINANCE_TRADFI = {"SPY": "SPYUSDT", "QQQ": "QQQUSDT", "GLD": "XAUUSDT", "SLV": "XAGUSDT"}
# Funding longs paid on those perps, Jan-Aug 2026, per year of notional (data.binance.vision fundingRate files).
# Holding a perp costs funding; the ETF returns above do not include it.
FUNDING_2026 = {"GLD": 0.121, "SLV": 0.230, "SPY": -0.061, "QQQ": 0.002}
TARGET_VOL = 0.12
COST = 0.0005


def etf_prices(start: str = "2006-01-01") -> pd.DataFrame:
    import yfinance as yf
    px = yf.download(ETFS, start=start, progress=False, auto_adjust=True)["Close"]
    return px.dropna(how="all")


def tsmom_returns(px: pd.DataFrame, funding: dict = None) -> pd.Series:
    """funding: yearly cost per ETF column charged on the notional held (e.g. FUNDING_2026)."""
    ret = px.pct_change(fill_method=None)
    mom = px / px.shift(252) - 1
    vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    month_end = px.index.to_series().groupby(px.index.to_period("M")).transform("max") == px.index.to_series()
    n = px.notna().sum(axis=1).clip(lower=1)
    target = ((mom > 0).astype(float) * (TARGET_VOL / np.sqrt(n)).values[:, None] / vol).where(month_end)
    w = target.ffill().fillna(0.0).clip(upper=1.0)
    held = w.shift(1).fillna(0.0)                        # decided at the close, earns from the next day
    gross = (held * ret.fillna(0.0)).sum(axis=1)
    cost = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1) * COST
    if funding:
        cost = cost + (held * pd.Series(funding).reindex(held.columns).fillna(0.0) / 252).sum(axis=1)
    return gross - cost


def crypto_trend_returns() -> pd.Series:
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    market = load_market_bulk("4h", last_month, UNIVERSE, lambda s: available_months(s, "4h"))
    end = max(b["timestamp"] for m in market.values() for b in m["bars"]) + 1
    curve: list = []
    simulate(market, LIVE, _ms(START), end, curve_out=curve)
    eq = pd.Series(dict(curve))
    eq.index = pd.to_datetime(eq.index, unit="ms")
    return eq.resample("1D").last().dropna().pct_change().dropna()


def mixed_bootstrap(crypto: pd.Series, etf: pd.Series, w: pd.Series, scale: float, L: float,
                    n_paths: int = 2000, block: int = 30, seed: int = 1) -> dict:
    """36-month paths, 500 + 100/month: crypto months drawn from `crypto`, ETF months drawn independently from
    `etf` (its long history, bad years included). The books are nearly uncorrelated, so drawing them
    independently is a fair approximation."""
    rng = np.random.default_rng(seed)
    c, e = crypto.to_numpy(), etf.to_numpy()
    hits24 = hits36 = below = 0
    for _ in range(n_paths):
        cs = rng.integers(0, len(c) - block, size=36)
        es = rng.integers(0, len(e) - block, size=36)
        path = scale * L * (w.iloc[0] * np.concatenate([c[i:i + block] for i in cs]) +
                            w.iloc[1] * np.concatenate([e[i:i + block] for i in es]))
        eq, hit = 500.0, None
        for d, v in enumerate(path, 1):
            eq = max(eq * (1 + v), 0.0)
            if d % 30 == 0:
                eq += 100.0
            if hit is None and eq >= 10_000:
                hit = d
            if d == 720:
                below += eq < 500 + 2400
        hits24 += hit is not None and hit <= 720
        hits36 += hit is not None
    return {"10k<=24mo": hits24 / n_paths, "10k<=36mo": hits36 / n_paths, "P(below deposits at 24mo)": below / n_paths}


def main():
    px = etf_prices()
    stock = tsmom_returns(px)
    s = stats(stock["2007-01-01":])
    print(f"ETF trend book alone, 2007 .. now (trading days): CAGR {s['cagr'] * 252 / 365 if False else ((1 + stock['2007':]).prod() ** (252 / len(stock['2007':])) - 1):+.1%}, "
          f"Sharpe {stock['2007':].mean() / stock['2007':].std() * np.sqrt(252):+.2f}, "
          f"max DD {((1 + stock['2007':]).cumprod() / (1 + stock['2007':]).cumprod().cummax() - 1).min():+.1%}")
    yearly = (1 + stock["2007":]).groupby(stock["2007":].index.year).prod() - 1
    print("  by year: " + " ".join(f"{y}:{v:+.0%}" for y, v in yearly.items()))

    crypto = crypto_trend_returns()
    # Put the stock book on calendar days (0 return on weekends/holidays) to line up with crypto
    days = pd.date_range(crypto.index.min(), crypto.index.max(), freq="D")
    stock_d = stock.reindex(days).fillna(0.0)
    crypto_d = crypto.reindex(days).fillna(0.0)
    both = pd.DataFrame({"crypto trend (1% risk)": crypto_d, "ETF trend": stock_d})
    print(f"\nCorrelation of daily returns 2020-06 .. now: {both.corr().iloc[0, 1]:+.2f}")
    vol_is = both[START:"2024-06-30"].std()
    w = (1 / vol_is) / (1 / vol_is).sum()
    combo = (both * w).sum(axis=1)
    combo *= both["crypto trend (1% risk)"][START:"2024-06-30"].std() / combo[START:"2024-06-30"].std()
    print(f"Equal-risk weights (from 2020-06 .. 2024-06 volatility): " + ", ".join(f"{k} {v:.0%}" for k, v in w.items()))
    print("\nIn-sample 2020-06 .. 2024-06 | out-of-sample 2024-07 .. now")
    for name in both.columns:
        print(fmt(name, both[name]))
    print(fmt("COMBINED (scaled to crypto's vol)", combo))

    print("\n500 + 100/month, 36-month paths from 30-day blocks of 2024-07 .. now, scaled to each risk level:")
    rows = []
    for name, r in (("crypto trend", both["crypto trend (1% risk)"]), ("combined", combo)):
        for L in (1, 2, 3):
            rows.append({"book": name, "crypto-equivalent risk": f"{L}%", **bootstrap_odds(r["2024-07-01":], L)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # Stricter: ETF months drawn from 2007 .. now (2008, 2016-18 included), crypto months from 2024-07 .. now
    scale = both["crypto trend (1% risk)"][START:"2024-06-30"].std() / (both * w).sum(axis=1)[START:"2024-06-30"].std()
    for label, prices, fund in (("all 12 ETFs", px, None),
                                ("Binance TradFi only (SPY, QQQ, gold, silver)", px[list(BINANCE_TRADFI)], None),
                                ("Binance TradFi, with 2026 perp funding", px[list(BINANCE_TRADFI)], FUNDING_2026)):
        book = tsmom_returns(prices, fund)
        long_hist = book["2007-01-01":]
        cal = long_hist.reindex(pd.date_range(long_hist.index.min(), long_hist.index.max(), freq="D")).fillna(0.0)
        b = book["2007":]
        print(f"\n{label}: ETF book 2007 .. now Sharpe {b.mean() / b.std() * np.sqrt(252):+.2f}, "
              f"correlation with crypto {pd.concat([crypto_d, book.reindex(days).fillna(0.0)], axis=1).corr().iloc[0, 1]:+.2f}")
        rows = [{"crypto-equivalent risk": f"{L}%", **mixed_bootstrap(crypto_d["2024-07-01":], cal, w, scale, L)}
                for L in (1, 2, 3)]
        print("  stricter odds (ETF months from 2007 .. now):")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
