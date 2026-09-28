"""
Smart-money watch: paper-only forward test of the positioning edge (research, no real orders).

backend/positioning_lab.py found that, across the 30 most-traded Binance USDT perps, coins where Binance's top traders
(by position) lean more long than all accounts beat the coins where they lean more short: 2022-2026, rebalanced in
7 daily slices, Sharpe ~1.1 (2022-23) and ~1.5 (2024+) including funding, nearly uncorrelated with the plan. It was
the one survivor of ~70 ideas, so it needs out-of-sample evidence before real money. This collects it:

  once a day   after 00:05 UTC, from Binance public endpoints (no key): the top 30 USDT perps by 30-day quote volume
               (60+ days listed), and for each the hourly top-trader position ratio and all-account ratio of the last
               3 UTC days; signal = mean(log top ratio - log account ratio)
  portfolio    long the 6 highest, short the 6 lowest (1/12 of the book each); the book holds the average of the
               last 7 daily portfolios (7 slices, so no weekday luck)
  paper P&L    daily closes, real funding (longs pay positive rates), 0.1% per unit of turnover
  alerts       Telegram once a week (Sunday): the week's and total paper return, current longs / shorts
  files        data/smart_watch/state.json, history.csv; GET /api/smart_watch/status for the plan page

PremiumWatch (below) runs a second paper book the same way on the Coinbase premium (data/premium_watch/,
GET /api/premium_watch/status).
"""

import csv
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger("layaquant.smart_watch")

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
BASE = "https://fapi.binance.com"
TOP_N, SLICES, COST, CANDIDATES = 30, 7, 0.001, 60
DAY_MS = 86_400_000


def _get(path: str, params: dict):
    """GET a Binance futures path, or any full URL (Binance spot, Coinbase)."""
    for k in range(4):
        try:
            r = requests.get(path if path.startswith("http") else BASE + path, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                time.sleep(30 * (k + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    return None


class SmartWatch:
    NAME, TITLE, LEGS = "smart_watch", "🐋 Smart-money watch", 6

    def __init__(self, api: Callable[[str, dict], object] = _get, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(DATA_ROOT, self.NAME)
        self.state_path = os.path.join(self.data_dir, "state.json")
        self.history_path = os.path.join(self.data_dir, "history.csv")
        os.makedirs(self.data_dir, exist_ok=True)
        self.api = api
        self.state = {"last_day": None, "last_end_ms": None, "equity": 1.0, "slices": [], "weights": {}, "closes": {},
                      "history": [], "last_week_report": None, "started": None}
        if os.path.exists(self.state_path):
            try:
                self.state.update(json.load(open(self.state_path)))
            except Exception as e:                        # noqa: BLE001
                logger.warning(f"{self.NAME} state unreadable: {e}")
        self._thread: Optional[threading.Thread] = None

    # ---------- data ----------
    def _daily(self, sym: str, day_end_ms: int, n: int = 62) -> List[list]:
        """Completed daily klines ending at day_end_ms (exclusive)."""
        k = self.api("/fapi/v1/klines", {"symbol": sym, "interval": "1d", "endTime": day_end_ms - 1, "limit": n}) or []
        return [x for x in k if int(x[6]) < day_end_ms]

    def universe(self, day_end_ms: int) -> Dict[str, float]:
        """{symbol: last close} for the top 30 by trailing 30-day quote volume (60+ days of history)."""
        info = self.api("/fapi/v1/exchangeInfo", {}) or {}
        perps = {s["symbol"] for s in info.get("symbols", []) if s.get("quoteAsset") == "USDT"
                 and s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"}
        tick = self.api("/fapi/v1/ticker/24hr", {}) or []
        cands = sorted((t for t in tick if t.get("symbol") in perps), key=lambda t: -float(t.get("quoteVolume", 0)))
        vols, closes = {}, {}
        for t in cands[:CANDIDATES]:
            k = self._daily(t["symbol"], day_end_ms)
            if len(k) >= 60:
                vols[t["symbol"]] = float(np.mean([float(x[7]) for x in k[-31:-1]]))   # 30 days up to the prior day
                closes[t["symbol"]] = float(k[-1][4])
        top = sorted(vols, key=lambda s: -vols[s])[:TOP_N]
        return {s: closes[s] for s in top}

    def signal(self, sym: str, day_end_ms: int) -> Optional[float]:
        start = day_end_ms - 3 * DAY_MS
        p = {"symbol": sym, "period": "1h", "limit": 100, "startTime": start, "endTime": day_end_ms - 1}
        top = self.api("/futures/data/topLongShortPositionRatio", p) or []
        acct = self.api("/futures/data/globalLongShortAccountRatio", p) or []
        a = {int(x["timestamp"]): float(x["longShortRatio"]) for x in acct}
        gaps = [math.log(float(x["longShortRatio"])) - math.log(a[int(x["timestamp"])]) for x in top
                if int(x["timestamp"]) in a and float(x["longShortRatio"]) > 0 and a[int(x["timestamp"])] > 0
                and start <= int(x["timestamp"]) < day_end_ms]
        return float(np.mean(gaps)) if len(gaps) >= 24 else None

    def funding(self, sym: str, start_ms: int, end_ms: int) -> float:
        f = self.api("/fapi/v1/fundingRate", {"symbol": sym, "startTime": start_ms, "endTime": end_ms - 1, "limit": 100}) or []
        return float(sum(float(x["fundingRate"]) for x in f))

    def close_of(self, sym: str, day_end_ms: int) -> Optional[float]:
        k = self._daily(sym, day_end_ms, n=2)
        return float(k[-1][4]) if k else None

    # ---------- daily step ----------
    def run_day(self, day: Optional[str] = None):
        """day = the UTC date just completed (default: yesterday). Marks the book to that day's close, then forms
        that day's slice from its positioning data."""
        from backend.notifier import notifier
        now = datetime.now(timezone.utc)
        day = day or (now - timedelta(days=1)).strftime("%Y-%m-%d")
        end_ms = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000) + DAY_MS
        if self.state.get("last_end_ms") and self.state["last_end_ms"] >= end_ms:
            return                                            # already marked to this close
        prev_ms = self.state.get("last_end_ms") or end_ms - DAY_MS   # covers missed days (server was off)
        closes = self.universe(end_ms)
        if len(closes) < 2 * self.LEGS:
            logger.warning(f"{self.NAME}: Binance data unavailable, retrying later")
            return

        # 1 mark the held book from the previous close to this close, funding included
        w_old = self.state["weights"]
        pnl = 0.0
        for sym, w in w_old.items():
            c1 = closes.get(sym) or self.close_of(sym, end_ms)
            c0 = self.state["closes"].get(sym)
            if c0 and c1:
                pnl += w * (c1 / c0 - 1)
            pnl -= w * self.funding(sym, prev_ms, end_ms)

        # 2 today's slice
        sig = {s: v for s in closes if (v := self.signal(s, end_ms)) is not None}
        legs = self.LEGS
        if len(sig) >= 2 * legs:
            ranked = sorted(sig, key=sig.get)
            today = {**{s: 0.5 / legs for s in ranked[-legs:]}, **{s: -0.5 / legs for s in ranked[:legs]}}
        else:
            today = {}
        self.state["slices"] = (self.state["slices"] + [today])[-SLICES:]
        target: Dict[str, float] = {}
        for sl in self.state["slices"]:
            for s, w in sl.items():
                target[s] = target.get(s, 0.0) + w / SLICES
        turnover = sum(abs(target.get(s, 0.0) - w_old.get(s, 0.0)) for s in set(target) | set(w_old))
        pnl -= turnover * COST
        self.state["equity"] *= 1 + pnl
        new_closes = {}
        for s in target:
            new_closes[s] = closes.get(s) or self.close_of(s, end_ms)
        self.state.update({"weights": {s: w for s, w in target.items() if abs(w) > 1e-12},
                           "closes": {s: c for s, c in new_closes.items() if c}, "last_day": day, "last_end_ms": end_ms,
                           "started": self.state.get("started") or day})
        self.state["history"].append({"day": day, "equity": round(self.state["equity"], 6), "return": round(pnl, 6),
                                      "turnover": round(turnover, 4), "coins": len(sig),
                                      "longs": " ".join(s for s, w in today.items() if w > 0),
                                      "shorts": " ".join(s for s, w in today.items() if w < 0)})
        self._write_history()

        # 3 weekly message (Sunday's close)
        week = datetime.fromisoformat(day).strftime("%G-W%V")
        if datetime.fromisoformat(day).weekday() == 6 and self.state.get("last_week_report") != week:
            self.state["last_week_report"] = week
            h = self.state["history"][-7:]
            wk = float(np.prod([1 + x["return"] for x in h]) - 1)
            longs = " ".join(s.replace("USDT", "") for s, w in sorted(target.items(), key=lambda x: -x[1])[:legs] if w > 0)
            shorts = " ".join(s.replace("USDT", "") for s, w in sorted(target.items(), key=lambda x: x[1])[:legs] if w < 0)
            notifier.send(f"{self.TITLE} (paper): this week {100 * wk:+.1f}%, since {self.state['started']} "
                          f"{100 * (self.state['equity'] - 1):+.1f}%.\nLeaning long: {longs}\nLeaning short: {shorts}")
        self._save()

    def _save(self):
        json.dump(self.state, open(self.state_path, "w"), indent=1)

    def _write_history(self):
        with open(self.history_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["day", "equity", "return", "turnover", "coins", "longs", "shorts"])
            w.writeheader()
            w.writerows(self.state["history"])

    def score(self) -> dict:
        r = np.array([x["return"] for x in self.state["history"][1:]])
        if len(r) < 2:
            return {"days": int(len(r)), "total": self.state["equity"] - 1, "sharpe": 0.0}
        sharpe = float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0
        return {"days": int(len(r)), "total": self.state["equity"] - 1, "sharpe": sharpe}

    def status(self) -> dict:
        w = self.state["weights"]
        return {"last_check": self.state["last_day"], "started": self.state.get("started"), "score": self.score(),
                "longs": sorted((s for s in w if w[s] > 0), key=lambda s: -w[s]),
                "shorts": sorted((s for s in w if w[s] < 0), key=lambda s: w[s]),
                "history": self.state["history"][-60:],
                "file": os.path.relpath(self.history_path, os.path.dirname(DATA_ROOT))}

    # ---------- background loop ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            try:
                now = datetime.now(timezone.utc)
                yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
                if self.state.get("last_day") != yesterday and (now.hour, now.minute) >= (0, 5):
                    self.run_day(yesterday)
            except Exception as e:                        # noqa: BLE001
                logger.warning(f"{self.NAME} error: {e}")
            time.sleep(900)


class PremiumWatch(SmartWatch):
    """Paper forward test of the Coinbase premium (cb_premium_lab.py): same top-30 universe and 7 daily slices, coins
    that also trade on Coinbase in USD (no 1000x tickers); signal = 7-day mean of log(Coinbase close / Binance spot
    close); long the 4 highest, short the 4 lowest. It failed its 2022-23 backtest and worked from 2024, so it is
    watched here, not traded."""
    NAME, TITLE, LEGS = "premium_watch", "🇺🇸 Coinbase-premium watch", 4
    COINBASE = "https://api.exchange.coinbase.com"
    SPOT = "https://api.binance.com/api/v3/klines"

    def __init__(self, api: Callable[[str, dict], object] = _get, data_dir: Optional[str] = None):
        super().__init__(api, data_dir)
        self._bases, self._bases_day = set(), None

    def coinbase_bases(self) -> set:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._bases_day != today:
            p = self.api(f"{self.COINBASE}/products", {}) or []
            bases = {x["base_currency"] for x in p if x.get("quote_currency") == "USD" and not x.get("trading_disabled")}
            if bases:
                self._bases, self._bases_day = bases, today
        return self._bases

    def signal(self, sym: str, day_end_ms: int) -> Optional[float]:
        base = sym[:-4]
        if sym.startswith("1000") or base not in self.coinbase_bases():
            return None
        start = day_end_ms - 7 * DAY_MS
        iso = lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cb = self.api(f"{self.COINBASE}/products/{base}-USD/candles",
                      {"granularity": 86400, "start": iso(start), "end": iso(day_end_ms - DAY_MS)}) or []
        spot = self.api(self.SPOT, {"symbol": sym, "interval": "1d", "startTime": start, "endTime": day_end_ms - 1}) or []
        cb_close = {int(x[0]) * 1000: float(x[4]) for x in cb if isinstance(x, list)}
        logs = [math.log(cb_close[int(x[0])] / float(x[4])) for x in spot
                if int(x[0]) in cb_close and float(x[4]) > 0 and cb_close[int(x[0])] > 0 and start <= int(x[0]) < day_end_ms]
        return float(np.mean(logs)) if len(logs) >= 5 else None


smart_watch = SmartWatch()
premium_watch = PremiumWatch()
