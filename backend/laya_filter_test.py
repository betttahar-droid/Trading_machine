"""
Does the Laya model improve the trend strategy's entries?

Takes every historical entry of the live trend strategy (4h, 120-bar breakout, long-only) since
2020-03, shows Laya the state at the signal bar, and checks whether the trades it approves did
better than the ones it rejects. A real filter should beat randomly skipping the same number of
trades, in-sample and out-of-sample.

Two prompts:
- PROD:  exactly what server.py sends today (same state text, same default buy/sell/hold criteria).
- TREND: a prompt written for this strategy, asking whether the breakout will keep trending.

Indicators in the state text use the paper trader's definitions (SMA 10/30, RSI 14, volume vs
10-bar average), computed on the 4h bars.

Needs `pip install laya torch`. Usage:
    python -m backend.laya_filter_test
"""

import time
from typing import Dict, List

import numpy as np

from backend.backtest_trend import SPLIT, _ms
from backend.growth_study import LIVE, WINDOW_START, _date, load_market_bulk
from backend.trend_strategy import TrendParams, compute_features, entry_signal, exit_on_close, trail_stop

PROD_QUESTIONS = {"decision": {
    "type": "choice",
    "instructions": "Evaluate the quantitative market metrics and determine the highest-probability trading action.",
    "criteria": {
        "buy": "RSI < 35 or fast SMA crossing above slow SMA with high volume expansion.",
        "sell": "RSI > 65 or fast SMA crossing below slow SMA indicating bearish momentum.",
        "hold": "Market in sideways consolidation with neutral indicators.",
    }}}
TREND_QUESTIONS = {"continue": {
    "type": "noul",
    "instructions": "Price just broke above its 20-day high. Is this breakout likely to continue into a "
                    "sustained uptrend rather than fail and reverse?",
}}


def symbol_trades(m: dict, f: Dict[str, np.ndarray], p: TrendParams, start_ms: int) -> List[dict]:
    """One symbol's long trades under backtest_trend.simulate's rules; R = net P&L / initial risk."""
    bars, trades, pos, pending = m["bars"], [], None, None
    for i, bar in enumerate(bars):
        if bar["timestamp"] < start_ms:
            continue
        if pending == "exit" and pos:
            fill = bar["open"] * (1 - p.slippage)
            pos["pnl"] += fill - pos["entry"] - fill * p.taker_fee
            trades.append({**pos, "r": pos["pnl"] / pos["risk"]})
            pos = None
        elif pending and pending != "exit" and pos is None:
            fill = bar["open"] * (1 + p.slippage)
            stop = fill - p.stop_atr * pending["atr"]
            pos = {"signal_i": pending["i"], "ts": bars[pending["i"]]["timestamp"], "entry": fill, "stop": stop,
                   "risk": fill - stop, "extreme": fill, "pnl": -fill * p.taker_fee}
        pending = None
        if pos:
            if bar["low"] <= pos["stop"]:
                fill = min(bar["open"], pos["stop"]) * (1 - p.slippage)
                pos["pnl"] += fill - pos["entry"] - fill * p.taker_fee
                trades.append({**pos, "r": pos["pnl"] / pos["risk"]})
                pos = None
            else:
                pos["pnl"] -= m["funding"][i] * bar["close"]
        if pos:
            pos["extreme"] = max(pos["extreme"], f["close"][i])
            if not np.isnan(f["atr"][i]):
                pos["stop"] = trail_stop("LONG", pos["extreme"], f["atr"][i], pos["stop"], p)
            if exit_on_close(f, i, "LONG"):
                pending = "exit"
        elif entry_signal(f, i, p):
            pending = {"i": i, "atr": float(f["atr"][i])}
    return trades


def state_text(sym: str, bars: List[dict], i: int) -> str:
    """server.py's state template, with the paper trader's indicator definitions."""
    closes = np.array([b["close"] for b in bars[i - 30:i + 1]])
    vols = np.array([b["volume"] for b in bars[i - 10:i + 1]])
    diffs = np.diff(closes[-15:])
    gains, losses = diffs[diffs > 0].sum() / 14.0, -diffs[diffs < 0].sum() / 14.0
    rsi = 100.0 - 100.0 / (1.0 + gains / max(losses, 1e-6))
    sma_fast, sma_slow = closes[-10:].mean(), closes[-30:].mean()
    vol_spike = vols[-1] / max(vols[-10:].mean(), 1e-4)
    return (f"Instrument: {sym}\n"
            f"Current Price: {closes[-1]:.4f}\n"
            f"RSI (14): {rsi:.1f}\n"
            f"Fast SMA: {sma_fast:.4f}\n"
            f"Slow SMA: {sma_slow:.4f}\n"
            f"Volume Spike: {vol_spike:.2f}x\n"
            f"Market Regime: ranging")


def compare(label: str, r: np.ndarray, keep: np.ndarray, rng: np.random.Generator):
    """Kept vs skipped trades, and how often a random filter of the same size does as well."""
    k = int(keep.sum())
    if k == 0 or k == len(r):
        print(f"  {label:<10} kept {k}/{len(r)} trades: filter is all-or-nothing, nothing to compare")
        return
    rand = np.array([r[rng.choice(len(r), k, replace=False)].mean() for _ in range(10_000)])
    p = (rand >= r[keep].mean()).mean()
    print(f"  {label:<10} kept {k:3d}/{len(r):3d}  E[R] kept {r[keep].mean():+.3f} vs skipped {r[~keep].mean():+.3f} "
          f"| total R kept {r[keep].sum():+6.1f} of {r.sum():+6.1f} | random filter does as well {p:.0%} of the time")


def main():
    import laya

    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    market = load_market_bulk("4h", last_month)
    trades = []
    for sym, m in market.items():
        for t in symbol_trades(m, compute_features(m["bars"], LIVE), LIVE, _ms(WINDOW_START)):
            trades.append({**t, "sym": sym})
    trades.sort(key=lambda t: t["ts"])
    r = np.array([t["r"] for t in trades])
    oos = np.array([t["ts"] >= _ms(SPLIT) for t in trades])
    print(f"{len(trades)} trend trades {_date(trades[0]['ts'])} .. {_date(trades[-1]['ts'])}: "
          f"win {np.mean(r > 0):.0%}, E[R] {r.mean():+.3f} (in-sample {r[~oos].mean():+.3f}, out-of-sample {r[oos].mean():+.3f})")

    agent = laya.load("convaiinnovations/laya", device="cpu")
    buy, p_up, t0 = [], [], time.time()
    for n, t in enumerate(trades):
        text = state_text(t["sym"], market[t["sym"]]["bars"], t["signal_i"])
        buy.append(agent.system_one(text, PROD_QUESTIONS)["answers"]["decision"]["choice"] == "buy")
        p_up.append(agent.system_one(text, TREND_QUESTIONS)["answers"]["continue"]["noul"])
        if n % 100 == 99:
            print(f"  ... {n + 1}/{len(trades)} scored ({time.time() - t0:.0f}s)", flush=True)
    buy, p_up = np.array(buy), np.array(p_up)

    rng = np.random.default_rng(0)
    print("\nPROD prompt (what server.py sends): take the trade only if Laya says 'buy'")
    for name, mask in (("all", np.ones(len(r), bool)), ("in-sample", ~oos), ("OOS", oos)):
        compare(name, r[mask], buy[mask], rng)

    cut = np.median(p_up[~oos])   # threshold fixed on in-sample trades only
    print(f"\nTREND prompt: take the trade only if P(breakout continues) >= {cut:.3f} (in-sample median)")
    print(f"  P(continue) range {p_up.min():.3f} .. {p_up.max():.3f}; "
          f"correlation with R: {np.corrcoef(p_up, r)[0, 1]:+.3f}")
    for name, mask in (("all", np.ones(len(r), bool)), ("in-sample", ~oos), ("OOS", oos)):
        compare(name, r[mask], p_up[mask] >= cut, rng)


if __name__ == "__main__":
    main()
