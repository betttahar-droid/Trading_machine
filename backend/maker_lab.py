"""
Limit (maker) orders for the live trend strategy instead of market (taker) orders.

backtest_trend.simulate fills entries and channel exits at the next bar's open as a taker (0.05% fee + 0.03%
slippage). Here they are limit orders at the signal bar's close, resting for one 4h bar:
  filled  if the next bar trades through the limit: maker fee 0.02%, no slippage, at the limit price
  missed  "chase": market order at the following open (taker); "skip" (entries only): no trade
Stops stay market orders (taker). Same 8 coins, same LIVE parameters, 2020-06 .. now; the question is whether fee
savings beat the trades that are missed or filled only when the breakout fails (adverse selection).

    python -m backend.maker_lab
"""

import time
from typing import Dict

import numpy as np
import pandas as pd

from backend.backtest_trend import UNIVERSE, _ms
from backend.growth_study import LIVE, load_market_bulk
from backend.strategy_lab import START
from backend.trend_strategy import compute_features, entry_signal, exit_on_close, position_size, trail_stop
from backend.universe_data import available_months

MAKER = 0.0002


def simulate(market: Dict[str, dict], p, start_ms: int, end_ms: int, mode: str, initial: float = 500.0):
    feats = {s: compute_features(m["bars"], p) for s, m in market.items()}
    timeline = sorted({b["timestamp"] for m in market.values() for b in m["bars"] if start_ms <= b["timestamp"] < end_ms})
    balance, positions, pending, curve = initial, {}, {}, []
    stats = {"fees": 0.0, "entries": 0, "maker_entries": 0, "missed": 0}

    def close(sym, pos, px, ts, taker=True):
        nonlocal balance
        fill = px * ((1 - p.slippage) if pos["side"] == "LONG" else (1 + p.slippage)) if taker else px
        gross = (fill - pos["entry"]) * pos["units"] * (1 if pos["side"] == "LONG" else -1)
        fee = abs(pos["units"]) * fill * (p.taker_fee if taker else MAKER)
        balance += gross - fee
        stats["fees"] += fee
        del positions[sym]

    def open_pos(sym, side, fill, atr, ts, taker=True):
        nonlocal balance
        gross_used = sum(abs(x["units"]) * x["mark"] for x in positions.values())
        equity = balance + sum(x["upnl"] for x in positions.values())
        notional = position_size(equity, fill, atr, gross_used, p, side)
        if notional < 10.0:
            return
        units = notional / fill
        stop = fill - p.stop_atr * atr if side == "LONG" else fill + p.stop_atr * atr
        fee = notional * (p.taker_fee if taker else MAKER)
        balance -= fee
        stats["fees"] += fee
        stats["entries"] += 1
        stats["maker_entries"] += (not taker)
        positions[sym] = {"side": side, "entry": fill, "units": units, "stop": stop, "extreme": fill,
                          "upnl": 0.0, "mark": fill}

    for ts in timeline:
        for sym in list(pending):
            m = market[sym]
            i = m["idx"].get(ts)
            if i is None:
                continue
            o = pending[sym]
            bar = m["bars"][i]
            if mode == "taker" or o.get("chase"):
                pending.pop(sym)
                if o["type"] == "exit" and sym in positions:
                    close(sym, positions[sym], bar["open"], ts, taker=True)
                elif o["type"] == "entry" and sym not in positions:
                    fill = bar["open"] * (1 + p.slippage if o["side"] == "LONG" else 1 - p.slippage)
                    open_pos(sym, o["side"], fill, o["atr"], ts, taker=True)
                continue
            lim = o["limit"]
            if o["type"] == "exit" and sym in positions:
                long = positions[sym]["side"] == "LONG"
                hit = bar["high"] >= lim if long else bar["low"] <= lim
                if hit:
                    pending.pop(sym)
                    close(sym, positions[sym], lim, ts, taker=False)
                else:
                    o["chase"] = True
            elif o["type"] == "entry" and sym not in positions:
                long = o["side"] == "LONG"
                hit = bar["low"] <= lim if long else bar["high"] >= lim
                if hit:
                    pending.pop(sym)
                    open_pos(sym, o["side"], lim, o["atr"], ts, taker=False)
                elif mode == "limit_skip":
                    pending.pop(sym)
                    stats["missed"] += 1
                else:
                    o["chase"] = True
            else:
                pending.pop(sym)

        for sym in list(positions):
            m = market[sym]
            i = m["idx"].get(ts)
            if i is None:
                continue
            pos, bar = positions[sym], m["bars"][i]
            if pos["side"] == "LONG" and bar["low"] <= pos["stop"]:
                close(sym, pos, min(bar["open"], pos["stop"]), ts)
                continue
            if pos["side"] == "SHORT" and bar["high"] >= pos["stop"]:
                close(sym, pos, max(bar["open"], pos["stop"]), ts)
                continue
            balance -= m["funding"][i] * pos["units"] * bar["close"] * (1 if pos["side"] == "LONG" else -1)

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
                if exit_on_close(f, i, pos["side"]) and sym not in pending:
                    pending[sym] = {"type": "exit", "limit": px}
            elif sym not in pending:
                sig = entry_signal(f, i, p)
                if sig:
                    pending[sym] = {"type": "entry", "side": sig, "atr": float(f["atr"][i]), "limit": px}
        curve.append((ts, balance + sum(x["upnl"] for x in positions.values())))
    return curve, stats


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    market = load_market_bulk("4h", last_month, UNIVERSE, lambda s: available_months(s, "4h"))
    end = max(b["timestamp"] for m in market.values() for b in m["bars"]) + 1
    for mode in ("taker", "limit_chase", "limit_skip"):
        curve, st = simulate(market, LIVE, _ms(START), end, mode)
        eq = pd.Series(dict(curve))
        eq.index = pd.to_datetime(eq.index, unit="ms")
        r = eq.resample("1D").last().pct_change().dropna()
        out = []
        for lab, x in (("2020-06..2024-06", r[:"2024-06-30"]), ("2024-07..now", r["2024-07-01":])):
            e = (1 + x).cumprod()
            out.append(f"{lab} {e.iloc[-1] ** (365 / len(x)) - 1:+.1%}/yr Sharpe {x.mean() / x.std() * np.sqrt(365):+.2f} "
                       f"DD {(e / e.cummax() - 1).min():+.0%}")
        print(f"{mode:12s} entries {st['entries']:4d} (maker {st['maker_entries']:4d}, missed {st['missed']:3d}), "
              f"fees {st['fees']:8.0f} | " + " | ".join(out))


if __name__ == "__main__":
    main()
