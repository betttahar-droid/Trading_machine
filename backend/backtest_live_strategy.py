"""
Honest backtest of the LIVE scanner strategy (paper_trader.evaluate_step + dual_model_engine).

Unlike walk_forward_5m_futures.py (which tests a different neural policy), this replays the
exact entry logic the live bot uses, over real Binance USD-M futures 5m klines, with:
- Signals computed on CLOSED 5m bars only; entry at the NEXT bar's open (no lookahead).
- HTF regime from CLOSED 1h bars only.
- Conservative intrabar ordering: if a stop and a target are both inside one bar, the stop wins.
- Realistic fees: maker for entry/limit targets, taker + slippage for stops/timeouts.
- News, L2 depth and Fear&Greed are unavailable historically -> neutral (score 0, depth 1.0, F&G 50).

Usage:
    python -m backend.backtest_live_strategy --days 90
    python -m backend.backtest_live_strategy --days 90 --compare
"""

import os
import json
import time
import argparse
from dataclasses import dataclass, replace, asdict
from typing import Dict, List, Optional

import numpy as np
import requests

from backend.dual_model_engine import dual_model_engine

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CACHE_DIR = os.path.join(DATA_DIR, "backtest_cache")
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT", "SUIUSDT"]
BAR_MS = 5 * 60 * 1000
HOUR_MS = 60 * 60 * 1000


@dataclass
class Params:
    # Entry filter (mirrors paper_trader scanner filter)
    min_score: float = 88.0
    range_min_score: float = 92.0
    allow_range: bool = True
    min_adx: float = 0.0             # 0 disables the ADX trend-strength filter (1h ADX)
    # Exits (mirrors paper_trader defaults)
    sl_atr: float = 1.35
    min_stop_pct: float = 0.008      # live floor: stop_dist = max(sl_atr*ATR, 0.8% of price)
    tp1_atr: float = 1.4
    tp1_ratio: float = 0.75
    tp1_in_r: bool = False           # True -> TP1 placed at tp1_r * stop distance instead of tp1_atr * ATR
    tp1_r: float = 1.0
    runner_lock_atr: float = 0.2
    final_tp_r: float = 2.0          # target_to_stop_ratio
    trail_trigger_atr: float = 1.5
    trail_atr: float = 1.5
    max_hold_bars: int = 12          # 60 minutes
    # Portfolio
    max_positions: int = 3
    risk_pct: float = 0.02
    leverage: float = 5.0
    cooldown_bars: int = 2           # 600s global cooldown after any close
    # Costs
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    slippage: float = 0.0002
    maker_entry: bool = True


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{start_ms}_{end_ms}.json")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)

    out, cursor = [], start_ms
    while cursor < end_ms:
        for attempt in range(4):
            try:
                resp = requests.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1500},
                    timeout=10,
                )
                resp.raise_for_status()
                raw = resp.json()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        if not raw:
            break
        for k in raw:
            vol, taker_buy = float(k[5]), float(k[9])
            out.append({
                "timestamp": int(k[0]), "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                "close": float(k[4]), "volume": vol, "cvd_delta": taker_buy - (vol - taker_buy),
            })
        cursor = int(raw[-1][0]) + 1
        time.sleep(0.15)

    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


# ----------------------------------------------------------------------------------------
# Indicators (identical math to PaperTrader._compute_atr_and_indicators)
# ----------------------------------------------------------------------------------------
def compute_indicators(candles: List[dict]):
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]
    cvd = [c.get("cvd_delta", 0.0) for c in candles]
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])) for i in range(1, len(candles))]
    atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else closes[-1] * 0.015
    atr_avg28 = float(np.mean(tr[-28:])) if len(tr) >= 28 else atr_14
    atr_exp = atr_14 / max(atr_avg28, 1e-4)
    sma_fast = float(np.mean(closes[-10:])) if len(closes) >= 10 else closes[-1]
    sma_slow = float(np.mean(closes[-30:])) if len(closes) >= 30 else closes[-1]
    if len(closes) >= 15:
        d = np.diff(closes[-15:])
        g, l = d[d > 0].sum() / 14.0, -d[d < 0].sum() / 14.0
        rsi = float(100.0 - 100.0 / (1.0 + g / max(l, 1e-6)))
    else:
        rsi = 50.0
    avg_vol = float(np.mean(volumes[-10:])) if len(volumes) >= 10 else volumes[-1]
    return atr_14, atr_exp, sma_fast, sma_slow, rsi, volumes[-1] / max(avg_vol, 1e-4), float(np.sum(cvd[-3:]))


def classify_regime(htf: List[dict]) -> str:
    if len(htf) < 15:
        return "RANGE_BOUND"
    _, atr_exp, sma_fast, sma_slow, _, _, _ = compute_indicators(htf)
    spread = (sma_fast - sma_slow) / max(sma_slow, 1e-6) * 100.0
    last = htf[-1]["close"]
    if atr_exp > 1.6 and abs(spread) < 0.4:
        return "TOXIC_CHOP"
    if spread > 0.5 and last >= sma_fast:
        return "TREND_BULL"
    if spread < -0.5 and last <= sma_fast:
        return "TREND_BEAR"
    return "RANGE_BOUND"


def adx(htf: List[dict], n: int = 14) -> float:
    if len(htf) < 2 * n + 1:
        return 0.0
    h = np.array([c["high"] for c in htf]); l = np.array([c["low"] for c in htf]); c = np.array([c["close"] for c in htf])
    up, dn = h[1:] - h[:-1], l[:-1] - l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h[1:] - l[1:], abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])])

    def wilder(x):
        s = np.empty_like(x); s[n - 1] = x[:n].sum()
        for i in range(n, len(x)):
            s[i] = s[i - 1] - s[i - 1] / n + x[i]
        return s[n - 1:]

    atr_s, p_s, n_s = wilder(tr), wilder(pdm), wilder(ndm)
    pdi, ndi = 100 * p_s / np.maximum(atr_s, 1e-12), 100 * n_s / np.maximum(atr_s, 1e-12)
    dx = 100 * abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-12)
    return float(np.mean(dx[-n:]))


# ----------------------------------------------------------------------------------------
# Signal generation (precomputed once per symbol, reused across parameter sets)
# ----------------------------------------------------------------------------------------
def build_signals(ltf: List[dict], htf: List[dict]) -> Dict[int, dict]:
    """Returns {bar_index: signal} where the signal is known at the CLOSE of bar_index."""
    signals = {}
    h_idx = 0
    neutral_fng = {"value": 50, "value_classification": "Neutral"}
    for i in range(30, len(ltf) - 1):
        close_time = ltf[i]["timestamp"] + BAR_MS
        while h_idx < len(htf) and htf[h_idx]["timestamp"] + HOUR_MS <= close_time:
            h_idx += 1
        htf_closed = htf[max(0, h_idx - 35):h_idx]
        window = ltf[i - 29:i + 1]
        atr_14, atr_exp, sma_f, sma_s, rsi, _, cvd_mom = compute_indicators(window)
        spread_pct = (sma_f - sma_s) / max(sma_s, 1e-6) * 100.0
        # Model B needs spread & CVD agreement; skip the full synthesis when it can't fire
        if not ((spread_pct > 0.06 and cvd_mom > 0) or (spread_pct < -0.06 and cvd_mom < 0)):
            continue
        regime = classify_regime(htf_closed)
        res = dual_model_engine.evaluate_opportunity(
            symbol="", htf_regime=regime, candles_5m=window, atr_14=atr_14, cvd_delta=cvd_mom, rsi=rsi,
            spread_pct=spread_pct, news_score=0.0, atr_expansion=atr_exp, ticker_sentiment={"score": 0.0},
            depth_ratio=1.0, spread_bps=1.0, fear_and_greed=neutral_fng,
        )
        if not res["dual_consensus_achieved"]:
            continue
        signals[i] = {
            "side": "LONG" if res["final_action"] == "BUY_LONG" else "SHORT",
            "score": float(res["consensus_score"]),
            "regime": regime,
            "squeeze": bool(res["model_b_micro"].get("is_squeeze")),
            "atr": atr_14,
            "lev_cap": float(res["recommended_leverage"]),
            "adx": adx(htf_closed[-35:]) if len(htf_closed) >= 29 else 0.0,
        }
    return signals


def passes_filter(sig: dict, p: Params) -> bool:
    if sig["score"] < p.min_score:
        return False
    if sig["regime"] == "RANGE_BOUND":
        if not p.allow_range:
            return False
        if not (sig["squeeze"] or sig["score"] >= p.range_min_score):
            return False
    if p.min_adx > 0 and sig["adx"] < p.min_adx:
        return False
    return True


# ----------------------------------------------------------------------------------------
# Portfolio simulation
# ----------------------------------------------------------------------------------------
def simulate(data: Dict[str, dict], p: Params, start_bar: int, end_bar: int, initial: float = 500.0) -> dict:
    syms = list(data.keys())
    cash = initial
    positions: Dict[str, dict] = {}
    closed: List[dict] = []
    equity_curve = []
    last_close_bar = -10_000

    def close_units(pos, units, px, taker, reason, bar):
        nonlocal cash
        fill = px * (1 - p.slippage) if (taker and pos["side"] == "LONG") else px * (1 + p.slippage) if taker else px
        gross = (fill - pos["entry"]) * units if pos["side"] == "LONG" else (pos["entry"] - fill) * units
        fee = units * fill * (p.taker_fee if taker else p.maker_fee)
        frac = units / pos["units"]
        margin_part = pos["margin"] * frac
        cash += max(0.0, margin_part + gross - fee)
        pos["pnl"] += gross - fee
        pos["fees"] += fee
        pos["units"] -= units
        pos["margin"] -= margin_part
        pos["exits"].append(reason)
        pos["exit_bar"] = bar

    n_bars = min(len(d["ltf"]) for d in data.values())
    for t in range(start_bar, min(end_bar, n_bars)):
        # 1. Manage open positions on bar t
        for sym in list(positions):
            pos = positions[sym]
            c = data[sym]["ltf"][t]
            long = pos["side"] == "LONG"
            hit_stop = c["low"] <= pos["stop"] if long else c["high"] >= pos["stop"]
            if hit_stop:  # conservative: stop first
                gap_px = min(c["open"], pos["stop"]) if long else max(c["open"], pos["stop"])
                close_units(pos, pos["units"], gap_px, True, "trail" if pos["tp1_hit"] else "stop", t)
            else:
                if not pos["tp1_hit"] and (c["high"] >= pos["tp1"] if long else c["low"] <= pos["tp1"]):
                    close_units(pos, pos["units"] * p.tp1_ratio, pos["tp1"], False, "tp1", t)
                    pos["tp1_hit"] = True
                    lock = pos["entry"] + p.runner_lock_atr * pos["atr"] * (1 if long else -1)
                    pos["stop"] = max(pos["stop"], lock) if long else min(pos["stop"], lock)
                if pos["units"] > 1e-12 and (c["high"] >= pos["tp"] if long else c["low"] <= pos["tp"]):
                    close_units(pos, pos["units"], pos["tp"], False, "tp", t)
                elif pos["units"] > 1e-12:
                    # trailing ratchet updated with this bar's extreme (applies from next bar)
                    if long:
                        pos["peak"] = max(pos["peak"], c["high"])
                        if pos["peak"] >= pos["entry"] + p.trail_trigger_atr * pos["atr"]:
                            pos["stop"] = max(pos["stop"], pos["entry"] * 1.002, pos["peak"] - p.trail_atr * pos["atr"])
                    else:
                        pos["peak"] = min(pos["peak"], c["low"])
                        if pos["peak"] <= pos["entry"] - p.trail_trigger_atr * pos["atr"]:
                            pos["stop"] = min(pos["stop"], pos["entry"] * 0.998, pos["peak"] + p.trail_atr * pos["atr"])
                    if not pos["tp1_hit"] and t - pos["entry_bar"] + 1 >= p.max_hold_bars:
                        close_units(pos, pos["units"], c["close"], True, "timeout", t)
            if pos["units"] <= 1e-12:
                pos["r"] = pos["pnl"] / pos["risk_usd"]
                closed.append(pos)
                del positions[sym]
                last_close_bar = t

        # 2. New entries: signal known at close of bar t, fill at open of bar t+1
        slots = p.max_positions - len(positions)
        if slots > 0 and t - last_close_bar >= p.cooldown_bars and t + 1 < n_bars:
            cands = []
            for sym in syms:
                if sym in positions:
                    continue
                sig = data[sym]["signals"].get(t)
                if sig and passes_filter(sig, p):
                    cands.append((sig["score"], sym, sig))
            if cands:
                cands.sort(key=lambda x: -x[0])
                _, sym, sig = cands[0]  # live bot opens at most one position per cycle
                nxt = data[sym]["ltf"][t + 1]
                long = sig["side"] == "LONG"
                entry = nxt["open"] if p.maker_entry else nxt["open"] * (1 + p.slippage if long else 1 - p.slippage)
                atr = sig["atr"]
                stop_dist = max(p.sl_atr * atr, entry * p.min_stop_pct)
                equity = cash + sum(x["margin"] for x in positions.values())
                risk_usd = equity * p.risk_pct
                lev = min(p.leverage, sig["lev_cap"])
                notional = min(risk_usd / (stop_dist / entry), cash * 0.80 * lev / max(1, slots))
                margin = notional / lev
                if margin >= 15.0 and cash >= margin:
                    units = notional / entry
                    fee = notional * (p.maker_fee if p.maker_entry else p.taker_fee)
                    cash -= margin + fee
                    sgn = 1 if long else -1
                    tp1_dist = p.tp1_r * stop_dist if p.tp1_in_r else p.tp1_atr * atr
                    positions[sym] = {
                        "symbol": sym, "side": sig["side"], "entry": entry, "units": units, "margin": margin,
                        "atr": atr, "stop": entry - sgn * stop_dist, "tp1": entry + sgn * tp1_dist,
                        "tp": entry + sgn * p.final_tp_r * stop_dist, "tp1_hit": False, "peak": entry,
                        "entry_bar": t + 1, "entry_ts": nxt["timestamp"], "pnl": -fee, "fees": fee,
                        "risk_usd": units * stop_dist, "exits": [], "regime": sig["regime"],
                        "score": sig["score"], "stop_pct": stop_dist / entry * 100, "tp1_pct": tp1_dist / entry * 100,
                    }
        equity_curve.append(cash + sum(x["margin"] for x in positions.values()))

    return report(closed, equity_curve, initial)


def report(closed: List[dict], curve: List[float], initial: float) -> dict:
    if not closed:
        return {"trades": 0, "end_equity": curve[-1] if curve else initial}
    r = np.array([t["r"] for t in closed])
    pnl = np.array([t["pnl"] for t in closed])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = np.array(curve)
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    exits = {}
    for t in closed:
        key = "+".join(t["exits"])
        exits[key] = exits.get(key, 0) + 1
    return {
        "trades": len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1),
        "avg_win_usd": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_usd": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "expectancy_R": round(float(r.mean()), 3),
        "profit_factor": round(float(wins.sum() / max(1e-9, -losses.sum())), 2),
        "fees_usd": round(sum(t["fees"] for t in closed), 2),
        "end_equity": round(float(eq[-1]), 2),
        "return_pct": round((eq[-1] / initial - 1) * 100, 1),
        "max_dd_pct": round(float(dd), 1),
        "avg_stop_pct": round(float(np.mean([t["stop_pct"] for t in closed])), 3),
        "avg_tp1_pct": round(float(np.mean([t["tp1_pct"] for t in closed])), 3),
        "exit_mix": dict(sorted(exits.items(), key=lambda kv: -kv[1])[:6]),
    }


def load_data(days: int, symbols: List[str]) -> Dict[str, dict]:
    end_ms = (int(time.time() * 1000) // (24 * HOUR_MS)) * (24 * HOUR_MS)  # UTC midnight: stable cache key
    start_ms = end_ms - days * 24 * HOUR_MS
    data = {}
    for sym in symbols:
        ltf = fetch_klines(sym, "5m", start_ms, end_ms)
        htf = fetch_klines(sym, "1h", start_ms - 40 * HOUR_MS, end_ms)
        data[sym] = {"ltf": ltf, "htf": htf}
    # align all symbols on common timestamps
    common = set.intersection(*[{c["timestamp"] for c in d["ltf"]} for d in data.values()])
    # signal cache is invalidated whenever the live signal code changes
    engine_src = os.path.join(os.path.dirname(__file__), "dual_model_engine.py")
    engine_ver = int(os.path.getmtime(engine_src))
    for sym, d in data.items():
        d["ltf"] = [c for c in d["ltf"] if c["timestamp"] in common]
        sig_cache = os.path.join(CACHE_DIR, f"signals_{sym}_{start_ms}_{end_ms}_{len(d['ltf'])}_{engine_ver}.json")
        if os.path.exists(sig_cache):
            with open(sig_cache, "r", encoding="utf-8") as f:
                d["signals"] = {int(k): v for k, v in json.load(f).items()}
        else:
            d["signals"] = build_signals(d["ltf"], d["htf"])
            with open(sig_cache, "w", encoding="utf-8") as f:
                json.dump(d["signals"], f)
        print(f"  {sym}: {len(d['ltf'])} bars, {len(d['signals'])} raw consensus signals")
    return data


VARIANTS = {
    "LIVE (current config)": Params(),
    "Taker entries (no maker fill assumption)": Params(maker_entry=False),
    "TP1 at 1.0R (not 1.4 ATR)": Params(tp1_in_r=True, tp1_r=1.0),
    "TP1 1.0R, 50% scale-out": Params(tp1_in_r=True, tp1_r=1.0, tp1_ratio=0.5),
    "No stop floor (pure 1.35 ATR)": Params(min_stop_pct=0.0),
    "Trend only (no RANGE entries)": Params(allow_range=False),
    "Trend only + ADX>=25": Params(allow_range=False, min_adx=25.0),
    "Trend+ADX25, TP1 1R 50%, hold 2h": Params(allow_range=False, min_adx=25.0, tp1_in_r=True, tp1_r=1.0,
                                             tp1_ratio=0.5, max_hold_bars=24),
    "Trend+ADX25, SL 2ATR, TP1 1R 50%, final 3R, hold 4h": Params(
        allow_range=False, min_adx=25.0, sl_atr=2.0, min_stop_pct=0.0, tp1_in_r=True, tp1_r=1.0,
        tp1_ratio=0.5, final_tp_r=3.0, max_hold_bars=48, trail_trigger_atr=2.0, trail_atr=2.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--compare", action="store_true", help="run all variants")
    args = ap.parse_args()

    print(f"Loading {args.days} days of Binance USD-M 5m data for {len(UNIVERSE)} symbols...")
    data = load_data(args.days, UNIVERSE)
    n = min(len(d["ltf"]) for d in data.values())
    split = int(n * 0.6)

    variants = VARIANTS if args.compare else {"LIVE (current config)": Params()}
    results = {}
    for name, p in variants.items():
        is_res = simulate(data, p, 31, split)
        oos_res = simulate(data, p, split, n)
        full = simulate(data, p, 31, n)
        results[name] = {"params": asdict(p), "in_sample": is_res, "out_of_sample": oos_res, "full": full}
        print(f"\n=== {name} ===")
        for label, res in (("IS  (first 60%)", is_res), ("OOS (last 40%) ", oos_res), ("FULL           ", full)):
            if res["trades"] == 0:
                print(f"  {label}: no trades")
                continue
            print(f"  {label}: trades={res['trades']:4d} win={res['win_rate_pct']:5.1f}% "
                  f"E[R]={res['expectancy_R']:+.3f} PF={res['profit_factor']:.2f} "
                  f"ret={res['return_pct']:+6.1f}% maxDD={res['max_dd_pct']:.1f}% fees=${res['fees_usd']:.0f} "
                  f"avgW=${res['avg_win_usd']:.2f} avgL=${res['avg_loss_usd']:.2f}")
        print(f"  stop≈{full.get('avg_stop_pct', 0)}%  tp1≈{full.get('avg_tp1_pct', 0)}%  exits={full.get('exit_mix')}")

    out = os.path.join(DATA_DIR, "backtest_live_strategy_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"days": args.days, "generated": time.strftime("%Y-%m-%d %H:%M"), "results": results}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
