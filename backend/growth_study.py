"""
Can the trend strategy turn $500 into $10,000 in months?

Replays the live trend rules (trend_strategy.py, the paper trader's 4h config) on Binance USD-M
4h klines + real funding history. A fresh $500 account is started every two weeks since 2020 and
run for 12 months, at increasing risk per trade. For each setting it reports how often the account
reached $10k within 3 / 6 / 12 months, and how often it was wiped out.

Fills match backtest_trend.simulate (entries and channel exits at the next open, resting stops
filled at the stop or the gap open, taker fee + slippage on every fill, funding at every
settlement). On top of that it liquidates the account (cross margin) when equity at the bar's
worst prices falls below maintenance margin, which matters once leverage is raised.

Klines and funding come from the data.binance.vision monthly archives (fapi.binance.com is
geo-blocked in some regions), cached under data/bulk_cache/.

Usage:
    python -m backend.growth_study
"""

import io
import os
import time
import zipfile
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import requests

from backend.backtest_trend import DATA_DIR, INTERVAL_MS, SPLIT, UNIVERSE, _ms, simulate
from backend.trend_strategy import TrendParams, compute_features, entry_signal, exit_on_close, trail_stop, position_size

BULK_URL = "https://data.binance.vision/data/futures/um/monthly"
BULK_CACHE = os.path.join(DATA_DIR, "bulk_cache")
FIRST_MONTH = "2020-01"

# The paper trader's live config (paper_trader.py: self.trend_params)
LIVE = TrendParams(interval="4h", entry_n=120, exit_n=60, stop_atr=4.0, allow_short=False, risk_pct=0.01)
# Leverage caps for the aggressive sweep: Binance allows this much on these coins
AGGRESSIVE_CAPS = dict(max_position_leverage=10.0, max_gross_leverage=20.0)
RISKS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]

START_EQUITY = 500.0
TARGET = 10_000.0
RUIN = 50.0                 # -90%: counted as wiped out
MAINT_MARGIN = 0.01         # liquidation when equity < 1% of open notional
WINDOW_START = "2020-03-01"
WINDOW_STEP_MS = 14 * 86_400_000
MONTH_MS = 365 * 86_400_000 // 12
HORIZON_MS = 12 * MONTH_MS


def _months(last: str) -> List[str]:
    y, m = map(int, FIRST_MONTH.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _bulk_csv(kind: str, symbol: str, month: str, interval: str) -> str:
    """One monthly archive as CSV text ('' when the symbol was not listed yet)."""
    name = f"{symbol}-{interval}-{month}" if kind == "klines" else f"{symbol}-fundingRate-{month}"
    path = os.path.join(BULK_CACHE, name + ".csv")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    sub = f"klines/{symbol}/{interval}" if kind == "klines" else f"fundingRate/{symbol}"
    for attempt in range(5):
        try:
            r = requests.get(f"{BULK_URL}/{sub}/{name}.zip", timeout=30)
            if r.status_code == 404:
                text = ""
            else:
                r.raise_for_status()
                z = zipfile.ZipFile(io.BytesIO(r.content))
                text = z.read(z.namelist()[0]).decode()
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    os.makedirs(BULK_CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def _rows(text: str) -> List[List[str]]:
    return [line.split(",") for line in text.splitlines() if line[:1].isdigit()]


def load_market_bulk(interval: str, last_month: str) -> Dict[str, dict]:
    """Same structure as backtest_trend.load_market, built from the monthly archives."""
    months = _months(last_month)
    market = {}
    for sym in UNIVERSE:
        with ThreadPoolExecutor(12) as ex:
            k_txt = list(ex.map(lambda m: _bulk_csv("klines", sym, m, interval), months))
            f_txt = list(ex.map(lambda m: _bulk_csv("funding", sym, m, interval), months))
        seen, bars = set(), []
        for text in k_txt:
            for r in _rows(text):
                ts = int(r[0])
                if ts not in seen:
                    seen.add(ts)
                    bars.append({"timestamp": ts, "open": float(r[1]), "high": float(r[2]),
                                 "low": float(r[3]), "close": float(r[4])})
        bars.sort(key=lambda b: b["timestamp"])
        if len(bars) < 150:
            continue
        funding = sorted((int(r[0]), float(r[-1])) for text in f_txt for r in _rows(text))
        step = INTERVAL_MS[interval]
        fund_per_bar = np.zeros(len(bars))
        j = 0
        for i, b in enumerate(bars):
            while j < len(funding) and funding[j][0] < b["timestamp"]:
                j += 1
            k = j
            while k < len(funding) and funding[k][0] < b["timestamp"] + step:
                fund_per_bar[i] += funding[k][1]
                k += 1
            j = k
        market[sym] = {"bars": bars, "funding": fund_per_bar, "idx": {b["timestamp"]: i for i, b in enumerate(bars)}}
    return market


def run_account(market: Dict[str, dict], feats: Dict[str, dict], timeline: List[int], p: TrendParams,
                start_ms: int, end_ms: int, liquidation: bool = True) -> dict:
    """backtest_trend.simulate's fill model for one long-only account, plus liquidation and target tracking."""
    balance = START_EQUITY
    positions: Dict[str, dict] = {}
    pending: Dict[str, dict] = {}
    hit_ts = ruin_ts = None
    peak, max_dd, n_trades = START_EQUITY, 0.0, 0
    cost = p.taker_fee

    def close(sym, pos, px):
        nonlocal balance, n_trades
        fill = px * (1 - p.slippage)
        balance += (fill - pos["entry"]) * pos["units"] - pos["units"] * fill * cost
        n_trades += 1
        del positions[sym]

    for ts in timeline[bisect_left(timeline, start_ms):bisect_left(timeline, end_ms)]:
        # 1. Opens: pending channel exits / entries
        for sym in list(pending):
            m = market[sym]
            i = m["idx"].get(ts)
            if i is None:
                continue
            order = pending.pop(sym)
            bar = m["bars"][i]
            if order["type"] == "exit" and sym in positions:
                close(sym, positions[sym], bar["open"])
            elif order["type"] == "entry" and sym not in positions:
                fill = bar["open"] * (1 + p.slippage)
                gross_used = sum(x["units"] * x["mark"] for x in positions.values())
                equity = balance + sum(x["upnl"] for x in positions.values())
                notional = position_size(equity, fill, order["atr"], gross_used, p)
                if notional < 10.0:
                    continue
                units = notional / fill
                balance -= notional * cost
                positions[sym] = {"entry": fill, "units": units, "stop": fill - p.stop_atr * order["atr"],
                                  "extreme": fill, "upnl": 0.0, "mark": fill}

        # 2. Intrabar: liquidation at the bar's worst prices (a stop caps a position's loss at its fill)
        if liquidation and positions:
            worst_eq, notional = balance, 0.0
            for sym, pos in positions.items():
                i = market[sym]["idx"].get(ts)
                if i is None:
                    worst = pos["mark"]
                else:
                    bar = market[sym]["bars"][i]
                    worst = min(bar["open"], pos["stop"]) * (1 - p.slippage) if bar["low"] <= pos["stop"] else bar["low"]
                worst_eq += (worst - pos["entry"]) * pos["units"]
                notional += worst * pos["units"]
            if worst_eq <= MAINT_MARGIN * notional:
                balance, positions = 0.0, {}
                ruin_ts = ruin_ts or ts
                max_dd = -1.0
                break

        # Resting stops, funding
        for sym in list(positions):
            m = market[sym]
            i = m["idx"].get(ts)
            if i is None:
                continue
            pos, bar = positions[sym], m["bars"][i]
            if bar["low"] <= pos["stop"]:
                close(sym, pos, min(bar["open"], pos["stop"]))
                continue
            balance -= m["funding"][i] * pos["units"] * bar["close"]

        # 3. Closes: mark, trail, queue exits and entries for the next open
        for sym, m in market.items():
            i = m["idx"].get(ts)
            if i is None:
                continue
            f = feats[sym]
            px = f["close"][i]
            if sym in positions:
                pos = positions[sym]
                pos["mark"] = px
                pos["upnl"] = (px - pos["entry"]) * pos["units"]
                pos["extreme"] = max(pos["extreme"], px)
                if not np.isnan(f["atr"][i]):
                    pos["stop"] = trail_stop("LONG", pos["extreme"], f["atr"][i], pos["stop"], p)
                if exit_on_close(f, i, "LONG"):
                    pending[sym] = {"type": "exit"}
            elif entry_signal(f, i, p):
                pending[sym] = {"type": "entry", "atr": float(f["atr"][i])}

        equity = balance + sum(x["upnl"] for x in positions.values())
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
        # The target counts only if closing everything right now still leaves >= $10k
        exit_cost = sum(x["units"] * x["mark"] for x in positions.values()) * (cost + p.slippage)
        if hit_ts is None and ruin_ts is None and equity - exit_cost >= TARGET:
            hit_ts = ts
        if ruin_ts is None and equity <= RUIN:
            ruin_ts = ts

    final = balance + sum(x["upnl"] for x in positions.values())
    return {"final": max(final, 0.0), "hit_ts": hit_ts, "ruin_ts": ruin_ts, "max_dd": max_dd, "trades": n_trades}


def _date(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    print(f"Loading 4h klines + funding {FIRST_MONTH} -> {last_month} from data.binance.vision ...", flush=True)
    market = load_market_bulk("4h", last_month)
    feats = {s: compute_features(m["bars"], LIVE) for s, m in market.items()}
    timeline = sorted({b["timestamp"] for m in market.values() for b in m["bars"]})
    end_ms = timeline[-1] + INTERVAL_MS["4h"]
    split = _ms(SPLIT)

    # Sanity check: the live config through backtest_trend.simulate, and run_account reproducing it.
    ref = simulate(market, LIVE, split, end_ms)
    mine = run_account(market, feats, timeline, LIVE, split, end_ms, liquidation=False)
    print(f"\nLive config, {SPLIT} -> {_date(end_ms)} (backtest_trend.simulate): CAGR {ref['cagr_pct']}%, "
          f"Sharpe {ref['sharpe']}, maxDD {ref['max_dd_pct']}%, end ${ref['end_equity']}  "
          f"| run_account end ${mine['final']:.2f}")

    grid = [(r, replace(LIVE, risk_pct=r, **AGGRESSIVE_CAPS)) for r in RISKS]

    for label, s0 in (("2020-03 -> now", _ms(WINDOW_START)), (f"{SPLIT} -> now (out of sample)", split)):
        years = (end_ms - s0) / (365 * 86_400_000)
        print(f"\n=== One $500 account, {label}, caps {AGGRESSIVE_CAPS} ===")
        print(f"{'risk':>6} {'end $':>14} {'CAGR':>9} {'maxDD':>7} {'trades':>7}  reached $10k")
        for r, p in grid:
            res = run_account(market, feats, timeline, p, s0, end_ms)
            cagr = (max(res["final"], 0.01) / START_EQUITY) ** (1 / years) - 1
            hit = _date(res["hit_ts"]) if res["hit_ts"] else "-"
            wiped = f"  WIPED OUT {_date(res['ruin_ts'])}" if res["ruin_ts"] else ""
            print(f"{r:6.1%} {res['final']:14,.0f} {cagr:9.0%} {res['max_dd']:7.0%} {res['trades']:7d}  {hit}{wiped}")

    starts = list(range(_ms(WINDOW_START), end_ms - HORIZON_MS, WINDOW_STEP_MS))
    years = sorted({_date(s)[:4] for s in starts})
    print(f"\n=== Fresh $500 account every 2 weeks, run 12 months: {len(starts)} starts "
          f"{_date(starts[0])} .. {_date(starts[-1])} ===")
    print(f"{'risk':>6} {'$10k<=3mo':>10} {'<=6mo':>7} {'<=12mo':>7} {'wiped':>7} {'<$250':>7} "
          f"{'median':>8} {'best':>9}   reached $10k within 12 months, by start year: " + " ".join(years))
    for r, p in grid:
        h3 = h6 = h12 = wiped = halved = 0
        finals, by_year = [], {y: [0, 0] for y in years}
        for s in starts:
            res = run_account(market, feats, timeline, p, s, s + HORIZON_MS)
            y = _date(s)[:4]
            by_year[y][1] += 1
            if res["hit_ts"] is not None:
                months = (res["hit_ts"] - s) / MONTH_MS
                h3 += months <= 3
                h6 += months <= 6
                h12 += 1
                by_year[y][0] += 1
            wiped += res["ruin_ts"] is not None and res["hit_ts"] is None
            halved += res["final"] < START_EQUITY / 2
            finals.append(res["final"])
        n = len(starts)
        cells = " ".join(f"{a}/{b}" for a, b in by_year.values())
        print(f"{r:6.1%} {h3 / n:10.0%} {h6 / n:7.0%} {h12 / n:7.0%} {wiped / n:7.0%} {halved / n:7.0%} "
              f"{np.median(finals):8,.0f} {max(finals):9,.0f}   {cells}", flush=True)


if __name__ == "__main__":
    main()
