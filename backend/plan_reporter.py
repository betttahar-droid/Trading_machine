"""
Local journal and Telegram messages for the trading plan.

Local record, on this PC, in data/journal/ (open with Excel):
  equity.csv   one row per hour: time, equity, deposited, trading P&L, cash, crypto / TradFi exposure, peak,
               drawdown, fees and funding paid since the plan started
  events.csv   every trade, deposit, TradFi rebalance, plan start / pause / resume, alert and error
Nothing is ever deleted: rows from earlier plans stay, and the plan page shows the current plan's rows.

Telegram (notifier.py): crypto entries and exits, one message per TradFi rebalance, deposits, a weekly summary
(Sunday from 18:00 UTC), an optional daily summary, drawdown alerts at 10/20/30/40% below the peak, target reached,
and errors (at most one per hour).
"""

import csv
import os
import time
from datetime import datetime, timezone
from typing import Dict, List

from backend.notifier import notifier

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
JOURNAL_DIR = os.path.join(DATA_DIR, "journal")
EQUITY_CSV = os.path.join(JOURNAL_DIR, "equity.csv")
EVENTS_CSV = os.path.join(JOURNAL_DIR, "events.csv")
EQUITY_COLS = ["time_utc", "ts", "equity", "deposited", "trading_pnl", "cash", "crypto_exposure", "tradfi_exposure",
               "peak", "drawdown_pct", "fees_total", "funding_total"]
EVENT_COLS = ["time_utc", "ts", "kind", "symbol", "amount", "text"]
DD_LEVELS = (10, 20, 30, 40)
HISTORICAL_WORST = {1.0: 25, 1.5: 35, 2.0: 45, 2.5: 50, 3.0: 60}   # combined plan, % below peak (backtests)
NAMES = {"XAUUSDT": "Gold", "XAGUSDT": "Silver", "SPYUSDT": "S&P 500", "QQQUSDT": "Nasdaq 100"}


def _append(path: str, cols: List[str], row: list):
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(cols)
        w.writerow(row)


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _money(v: float) -> str:
    return f"{v:,.2f}"


class PlanReporter:
    def __init__(self):
        self._throttle: Dict[str, float] = {}

    # ---------- helpers ----------
    @staticmethod
    def _state(pt) -> dict:
        return pt.plan.setdefault("report", {})

    def event(self, kind: str, text: str, symbol: str = "", amount=None, notify: bool = True):
        now = time.time()
        _append(EVENTS_CSV, EVENT_COLS, [_utc(now), int(now), kind, symbol, "" if amount is None else round(amount, 2), text])
        if notify:
            notifier.send(text)

    def _once_per(self, key: str, seconds: float) -> bool:
        now = time.time()
        if now - self._throttle.get(key, 0) < seconds:
            return False
        self._throttle[key] = now
        return True

    # ---------- hooks called by the paper trader ----------
    def on_plan(self, pt, kind: str):
        p = pt.plan
        if kind == "start":
            r = self._state(pt)
            eq = pt._trend_equity()
            r.update(peak=eq, week_start_equity=eq, day_start_equity=eq, deposits_week=0.0, trades_week=0,
                     dd_alerted=[], target_hit=False, fees_total=0.0, funding_total=0.0)
            text = (f"▶️ Plan started (paper): budget {_money(p['budget'])}, risk level {p['risk_level']:g}, "
                    f"deposit {_money(pt.monthly_deposit)}/month, target {_money(pt.equity_target)}. "
                    f"{'Crypto + TradFi books.' if p.get('tradfi') else 'Crypto book only.'}")
        elif kind == "pause":
            text = f"⏸ Plan paused. Equity {_money(pt._trend_equity())}. Positions stay open but are not watched."
        else:
            text = f"▶️ Plan resumed. Equity {_money(pt._trend_equity())}."
        self.event("plan", text)

    def on_trade(self, pt, rec: dict):
        if not pt.plan.get("active"):
            return
        r = self._state(pt)
        r["fees_total"] = r.get("fees_total", 0.0) + float(rec.get("fee_usd") or 0.0)
        side, sym = rec.get("side", ""), rec.get("symbol", "")
        coin = sym.replace("USDT", "")
        pnl = float(rec.get("pnl_usd") or 0.0)
        if side.startswith("TRADFI"):
            # one message per rebalance covers these; still recorded
            self.event("trade", f"TradFi {side.replace('TRADFI_', '').replace('_', ' ').lower()} {NAMES.get(sym, coin)} "
                                f"at {rec.get('price')} ({rec.get('reason', '')})", sym, pnl, notify=False)
            return
        r["trades_week"] = r.get("trades_week", 0) + 1
        if side in ("BUY_LONG", "SELL_SHORT"):
            text = f"🟢 Crypto entry: {coin} {'long' if side == 'BUY_LONG' else 'short'} at {rec.get('price')}. {rec.get('reason', '')}"
        else:
            icon = "✅" if pnl > 0 else "🔻"
            text = (f"{icon} Crypto exit: {coin} at {rec.get('price')}, P&L {pnl:+,.2f} ({rec.get('rrr', '')}). "
                    f"{rec.get('reason', '')}. Equity {_money(pt._trend_equity())}.")
        self.event("trade", text, sym, pnl)

    def on_funding(self, pt, amount: float):
        if pt.plan.get("active"):
            r = self._state(pt)
            r["funding_total"] = r.get("funding_total", 0.0) + amount

    def on_deposit(self, pt, amount: float):
        r = self._state(pt)
        r["peak"] = r.get("peak", 0.0) + amount           # a deposit is not a recovery from a drawdown
        r["deposits_week"] = r.get("deposits_week", 0.0) + amount
        self.event("deposit", f"💶 Deposit of {_money(amount)} credited. Total deposited {_money(pt.initial_capital)}, "
                              f"equity {_money(pt._trend_equity())}.", amount=amount)

    def on_rebalance(self, pt, weights: Dict[str, float], sig: Dict[str, dict]):
        from backend.tradfi_book import ASSETS
        lines = []
        for sym, w in weights.items():
            s = sig.get(ASSETS[sym], {})
            held = pt.positions.get(sym)
            lines.append(f"• {NAMES.get(sym, sym)}: 12-month {s.get('mom12', 0):+.1%} → "
                         f"{'hold ' + _money(held['units'] * held.get('current_price', held['entry_price'])) if held else 'flat'}"
                         f" ({w:.2f}× equity target)")
        self.event("rebalance", "📊 TradFi monthly rebalance\n" + "\n".join(lines))

    def on_warning(self, pt, key: str, text: str):
        if self._once_per("warn:" + key, 6 * 3600):
            self.event("warning", "⚠️ " + text)

    def on_error(self, pt, exc: Exception):
        if self._once_per("error", 3600):
            self.event("error", f"❗ Bot error (it keeps running and retries): {type(exc).__name__}: {exc}")

    # ---------- every loop ----------
    def tick(self, pt, now: float = None):
        if not pt.plan.get("active"):
            return
        now = now or time.time()
        r = self._state(pt)
        eq = pt._trend_equity()
        r["peak"] = max(r.get("peak", eq), eq)
        dd = (1 - eq / r["peak"]) * 100 if r["peak"] > 0 else 0.0

        hour = int(now // 3600)
        if r.get("last_hour") != hour:
            r["last_hour"] = hour
            exp = {"CRYPTO": 0.0, "TRADFI": 0.0}
            for p in pt.positions.values():
                book = "TRADFI" if p.get("strategy") == "TRADFI" else "CRYPTO"
                exp[book] += p["units"] * float(p.get("current_price", p["entry_price"]))
            _append(EQUITY_CSV, EQUITY_COLS, [
                _utc(now), int(now), round(eq, 2), round(pt.initial_capital, 2), round(eq - pt.initial_capital, 2),
                round(pt.cash, 2), round(exp["CRYPTO"] / max(eq, 1e-9), 3), round(exp["TRADFI"] / max(eq, 1e-9), 3),
                round(r["peak"], 2), round(dd, 2), round(r.get("fees_total", 0.0), 2), round(r.get("funding_total", 0.0), 2)])

        alerted = r.setdefault("dd_alerted", [])
        if eq >= r["peak"] and alerted:
            alerted.clear()
        crossed = [lv for lv in DD_LEVELS if dd >= lv and lv not in alerted]
        if crossed:                                          # one message, even if a big move crosses several levels
            alerted.extend(crossed)
            worst = HISTORICAL_WORST.get(float(pt.plan.get("risk_level", 2)), 45)
            advice = ("within the normal range for this risk level" if dd < worst else
                      "deeper than this risk level's historical worst: worth reviewing the plan")
            self.event("alert", f"📉 Drawdown alert: equity {_money(eq)} is {dd:.1f}% below its peak {_money(r['peak'])} "
                                f"({advice}; historical worst at risk level {pt.plan.get('risk_level', 2):g} "
                                f"is about −{worst}%).", amount=-dd)
        if eq >= pt.equity_target and not r.get("target_hit"):
            r["target_hit"] = True
            self.event("alert", f"🎯 Target reached: equity {_money(eq)} ≥ {_money(pt.equity_target)}.", amount=eq)

        day = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
        if r.get("last_day") != day:
            if r.get("last_day") and notifier.cfg.get("daily"):
                self._summary(pt, "📅 Daily summary", eq - r.get("day_start_equity", eq))
            r["last_day"], r["day_start_equity"] = day, eq

        utc = datetime.fromtimestamp(now, tz=timezone.utc)
        week = utc.strftime("%G-W%V")
        if utc.weekday() == 6 and utc.hour >= 18 and r.get("last_week") != week:
            if notifier.cfg.get("weekly"):
                change = eq - r.get("week_start_equity", eq) - r.get("deposits_week", 0.0)
                self._summary(pt, "🗓 Weekly summary", change)
            r.update(last_week=week, week_start_equity=eq, deposits_week=0.0, trades_week=0)

    def _summary(self, pt, title: str, change: float):
        r = self._state(pt)
        eq = pt._trend_equity()
        dep = pt.initial_capital
        held = []
        for sym, p in pt.positions.items():
            px = float(p.get("current_price", p["entry_price"]))
            up = (px - p["entry_price"]) * p["units"]
            held.append(f"{NAMES.get(sym, sym.replace('USDT', ''))} {up:+,.0f}")
        text = (f"{title}\n"
                f"Equity {_money(eq)} ({change:+,.2f} trading result this period)\n"
                f"Deposited {_money(dep)} · trading P&L {eq - dep:+,.2f} ({(eq / max(dep, 1e-9) - 1) * 100:+.1f}%)\n"
                f"Target {_money(pt.equity_target)}: {eq / max(pt.equity_target, 1e-9) * 100:.1f}% · drawdown "
                f"{(1 - eq / max(r.get('peak', eq), 1e-9)) * 100:.1f}%\n"
                f"Crypto trades this period: {r.get('trades_week', 0)} · fees {_money(r.get('fees_total', 0.0))} · "
                f"funding {_money(r.get('funding_total', 0.0))} since start\n"
                f"Open: {', '.join(held) if held else 'none'}")
        self.event("summary", text)

    # ---------- reading the journal ----------
    def journal(self, since_ts: float = 0, max_points: int = 600) -> dict:
        rows, events = [], []
        if os.path.exists(EQUITY_CSV):
            with open(EQUITY_CSV, encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if float(r["ts"]) >= since_ts]
        if len(rows) > max_points:
            step = len(rows) / max_points
            rows = [rows[int(i * step)] for i in range(max_points)] + [rows[-1]]
        if os.path.exists(EVENTS_CSV):
            with open(EVENTS_CSV, encoding="utf-8") as f:
                events = [e for e in csv.DictReader(f) if float(e["ts"]) >= since_ts][-60:][::-1]
        return {"equity": [{"t": int(float(r["ts"])), "equity": float(r["equity"]), "deposited": float(r["deposited"]),
                            "drawdown_pct": float(r["drawdown_pct"])} for r in rows],
                "events": events, "files": {"equity": EQUITY_CSV, "events": EVENTS_CSV}}


plan_reporter = PlanReporter()
