"""
Multi-year portfolio backtest of the trend-following rules in trend_strategy.py.

- Binance USD-M perpetual klines + the REAL historical funding-rate series per symbol.
- Signals on closed bars; entries/channel exits at the next bar's open; ATR stops are resting
  stop-market orders filled at the stop (or the open, if the bar gapped through it).
- Every fill pays taker fee + slippage. Funding is charged/credited on every 8h settlement held.
- In-sample / out-of-sample split by date. The PRIMARY config was fixed before any results were seen;
  the grid is shown only to judge robustness (a real edge should not depend on one exact setting).

Usage:
    python -m backend.backtest_trend                # primary config + robustness grid
    python -m backend.backtest_trend --no-grid
"""

import os
import json
import time
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import requests

from backend.trend_strategy import TrendParams, compute_features, entry_signal, exit_on_close, trail_stop, position_size

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CACHE_DIR = os.path.join(DATA_DIR, "backtest_cache")
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT", "SUIUSDT"]
INTERVAL_MS = {"1d": 86_400_000, "4h": 14_400_000}
START = "2020-01-01"
SPLIT = "2024-07-01"          # IS: START..SPLIT, OOS: SPLIT..now


def _ms(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _get(url: str, params: dict):
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def _cached(name: str, fn):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    data = fn()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[dict]:
    def load():
        out, cur = [], start_ms
        while cur < end_ms:
            raw = _get("https://fapi.binance.com/fapi/v1/klines",
                       {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms - 1, "limit": 1500})
            if not raw:
                break
            out += [{"timestamp": int(k[0]), "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                     "close": float(k[4])} for k in raw]
            cur = int(raw[-1][0]) + 1
            time.sleep(0.2)
        return out
    return _cached(f"trend_{symbol}_{interval}_{start_ms}_{end_ms}.json", load)


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> List[list]:
    def load():
        out, cur = [], start_ms
        while cur < end_ms:
            raw = _get("https://fapi.binance.com/fapi/v1/fundingRate",
                       {"symbol": symbol, "startTime": cur, "endTime": end_ms - 1, "limit": 1000})
            if not raw:
                break
            out += [[int(x["fundingTime"]), float(x["fundingRate"])] for x in raw]
            cur = int(raw[-1]["fundingTime"]) + 1
            time.sleep(0.2)
        return out
    return _cached(f"funding_{symbol}_{start_ms}_{end_ms}.json", load)


def load_market(interval: str, end_ms: int) -> Dict[str, dict]:
    start_ms = _ms(START)
    market = {}
    for sym in UNIVERSE:
        bars = fetch_klines(sym, interval, start_ms, end_ms)
        if len(bars) < 150:
            continue
        funding = fetch_funding(sym, start_ms, end_ms)
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


def simulate(market: Dict[str, dict], p: TrendParams, start_ms: int, end_ms: int, initial: float = 500.0,
             curve_out: list = None) -> dict:
    """curve_out, if given, receives (bar_ts, equity) at every close."""
    feats = {s: compute_features(m["bars"], p) for s, m in market.items()}
    timeline = sorted({b["timestamp"] for m in market.values() for b in m["bars"] if start_ms <= b["timestamp"] < end_ms})

    balance = initial
    positions: Dict[str, dict] = {}
    pending: Dict[str, dict] = {}     # orders to execute at the next bar's open
    trades, curve = [], []
    fees_total = funding_total = 0.0
    cost = p.taker_fee

    def close(sym, pos, px, ts, reason):
        nonlocal balance, fees_total
        fill = px * (1 - p.slippage) if pos["side"] == "LONG" else px * (1 + p.slippage)
        gross = (fill - pos["entry"]) * pos["units"] * (1 if pos["side"] == "LONG" else -1)
        fee = abs(pos["units"]) * fill * cost
        balance += gross - fee
        fees_total += fee
        pos["pnl"] += gross - fee
        trades.append({"symbol": sym, "side": pos["side"], "entry_ts": pos["entry_ts"], "exit_ts": ts,
                       "pnl": pos["pnl"], "r": pos["pnl"] / pos["risk_usd"], "reason": reason,
                       "bars": pos["bars"]})
        del positions[sym]

    for ts in timeline:
        # 1. Opens: execute pending exits / entries at this bar's open
        for sym in list(pending):
            m = market[sym]
            i = m["idx"].get(ts)
            if i is None:
                continue
            order = pending.pop(sym)
            bar = m["bars"][i]
            if order["type"] == "exit" and sym in positions:
                close(sym, positions[sym], bar["open"], ts, "channel")
            elif order["type"] == "entry" and sym not in positions:
                side = order["side"]
                fill = bar["open"] * (1 + p.slippage if side == "LONG" else 1 - p.slippage)
                gross_used = sum(abs(x["units"]) * x["mark"] for x in positions.values())
                equity = balance + sum(x["upnl"] for x in positions.values())
                notional = position_size(equity, fill, order["atr"], gross_used, p, side)
                if notional < 10.0:
                    continue
                units = notional / fill
                stop = fill - p.stop_atr * order["atr"] if side == "LONG" else fill + p.stop_atr * order["atr"]
                fee = notional * cost
                balance -= fee
                fees_total += fee
                positions[sym] = {"side": side, "entry": fill, "units": units, "stop": stop, "extreme": fill,
                                  "entry_ts": ts, "risk_usd": units * abs(fill - stop), "pnl": -fee,
                                  "upnl": 0.0, "mark": fill, "bars": 0}

        # 2. Intrabar: resting stops, funding
        for sym in list(positions):
            m = market[sym]
            i = m["idx"].get(ts)
            if i is None:
                continue
            pos, bar = positions[sym], m["bars"][i]
            pos["bars"] += 1
            if pos["side"] == "LONG" and bar["low"] <= pos["stop"]:
                close(sym, pos, min(bar["open"], pos["stop"]), ts, "stop")
                continue
            if pos["side"] == "SHORT" and bar["high"] >= pos["stop"]:
                close(sym, pos, max(bar["open"], pos["stop"]), ts, "stop")
                continue
            fund = m["funding"][i] * pos["units"] * bar["close"] * (1 if pos["side"] == "LONG" else -1)
            balance -= fund
            funding_total += fund
            pos["pnl"] -= fund

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
                pos["upnl"] = (px - pos["entry"]) * pos["units"] * (1 if pos["side"] == "LONG" else -1)
                pos["extreme"] = max(pos["extreme"], px) if pos["side"] == "LONG" else min(pos["extreme"], px)
                if not np.isnan(f["atr"][i]):
                    pos["stop"] = trail_stop(pos["side"], pos["extreme"], f["atr"][i], pos["stop"], p)
                if exit_on_close(f, i, pos["side"]):
                    pending[sym] = {"type": "exit"}
            else:
                sig = entry_signal(f, i, p)
                if sig:
                    pending[sym] = {"type": "entry", "side": sig, "atr": float(f["atr"][i])}

        curve.append((ts, balance + sum(x["upnl"] for x in positions.values())))

    if curve_out is not None:
        curve_out.extend(curve)
    return summarize(trades, curve, initial, fees_total, funding_total)


def summarize(trades, curve, initial, fees, funding) -> dict:
    if not curve:
        return {"trades": 0}
    days: Dict[str, float] = {}
    for ts, eq in curve:
        days[datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")] = eq
    eq = np.array(list(days.values()))
    rets = np.diff(eq) / eq[:-1]
    years = max(len(eq) / 365.0, 1e-9)
    by_year: Dict[str, List[float]] = {}
    for d, v in days.items():
        by_year.setdefault(d[:4], []).append(v)
    prev, yearly = initial, {}
    for y, vals in by_year.items():
        yearly[y] = round((vals[-1] / prev - 1) * 100, 1)
        prev = vals[-1]
    r = np.array([t["r"] for t in trades]) if trades else np.array([0.0])
    pnl = np.array([t["pnl"] for t in trades]) if trades else np.array([0.0])
    return {
        "trades": len(trades),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 1),
        "expectancy_R": round(float(r.mean()), 3),
        "avg_win_R": round(float(r[r > 0].mean()), 2) if (r > 0).any() else 0.0,
        "avg_loss_R": round(float(r[r <= 0].mean()), 2) if (r <= 0).any() else 0.0,
        "profit_factor": round(float(pnl[pnl > 0].sum() / max(1e-9, -pnl[pnl <= 0].sum())), 2),
        "end_equity": round(float(eq[-1]), 2),
        "cagr_pct": round(float(((eq[-1] / initial) ** (1 / years) - 1) * 100), 1),
        "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(365)), 2) if rets.std() > 0 else 0.0,
        "max_dd_pct": round(float((eq / np.maximum.accumulate(eq) - 1).min() * 100), 1),
        "fees_usd": round(fees, 2),
        "funding_usd": round(funding, 2),
        "yearly_pct": yearly,
        "avg_hold_bars": round(float(np.mean([t["bars"] for t in trades])), 1) if trades else 0.0,
    }


def buy_and_hold(market, start_ms, end_ms, initial=500.0) -> dict:
    bars = [b for b in market["BTCUSDT"]["bars"] if start_ms <= b["timestamp"] < end_ms]
    curve = [(b["timestamp"], initial * b["close"] / bars[0]["open"]) for b in bars]
    return summarize([], curve, initial, 0.0, 0.0)


def fmt(label, r):
    if not r.get("trades") and "sharpe" not in r:
        return f"  {label}: no data"
    return (f"  {label}: trades={r['trades']:4d} win={r['win_rate_pct']:5.1f}% E[R]={r['expectancy_R']:+.3f} "
            f"PF={r['profit_factor']:.2f} CAGR={r['cagr_pct']:+6.1f}% Sharpe={r['sharpe']:+.2f} "
            f"maxDD={r['max_dd_pct']:.1f}% fees=${r['fees_usd']:.0f} funding=${r['funding_usd']:.0f}")


PRIMARY = TrendParams()   # pre-registered: 1d, Donchian 20/10, 3x ATR(20) chandelier, 1% risk, long+short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()

    end_ms = (int(time.time() * 1000) // INTERVAL_MS["1d"]) * INTERVAL_MS["1d"]
    s0, split = _ms(START), _ms(SPLIT)
    markets = {}
    for iv in ("1d", "4h"):
        print(f"Loading {iv} klines + funding history {START} -> today ...", flush=True)
        markets[iv] = load_market(iv, end_ms)
    warm = s0 + 120 * INTERVAL_MS["1d"]   # indicator warm-up

    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "split": SPLIT, "results": {}}
    m1d = markets["1d"]
    print("\n=== Benchmark: BTC buy & hold ===")
    print(f"  IS : CAGR={buy_and_hold(m1d, warm, split)['cagr_pct']}%  maxDD={buy_and_hold(m1d, warm, split)['max_dd_pct']}%")
    print(f"  OOS: CAGR={buy_and_hold(m1d, split, end_ms)['cagr_pct']}%  maxDD={buy_and_hold(m1d, split, end_ms)['max_dd_pct']}%")

    def run(name, p):
        m = markets[p.interval]
        is_r, oos_r = simulate(m, p, warm, split), simulate(m, p, split, end_ms)
        out["results"][name] = {"params": asdict(p), "in_sample": is_r, "out_of_sample": oos_r}
        return is_r, oos_r

    is_r, oos_r = run("PRIMARY", PRIMARY)
    print("\n=== PRIMARY (pre-registered): 1d Donchian 20/10, 3xATR trail, 1% risk, long+short ===")
    print(fmt("IS  2020-24", is_r)); print(fmt("OOS 2024-26", oos_r))
    print(f"  yearly IS {is_r['yearly_pct']}  OOS {oos_r['yearly_pct']}")

    if not args.no_grid:
        print("\n=== Robustness grid (IS Sharpe -> OOS Sharpe | OOS E[R] | OOS maxDD) ===")
        for iv in ("1d", "4h"):
            for entry_n in ((20, 55, 100) if iv == "1d" else (30, 60, 120, 240)):
                for stop_atr in (2.0, 3.0, 4.0):
                    for short in (True, False):
                        p = replace(PRIMARY, interval=iv, entry_n=entry_n, exit_n=entry_n // 2,
                                    stop_atr=stop_atr, allow_short=short)
                        name = f"{iv} N={entry_n:<3d} stop={stop_atr:.0f}ATR {'L+S' if short else 'L  '}"
                        a, b = run(name, p)
                        print(f"  {name}: IS {a['sharpe']:+.2f} -> OOS {b['sharpe']:+.2f} | E[R] {b['expectancy_R']:+.3f} "
                              f"| DD {b['max_dd_pct']:.1f}% | trades {b['trades']}", flush=True)

    path = os.path.join(DATA_DIR, "backtest_trend_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
