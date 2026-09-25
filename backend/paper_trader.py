import os
import sys
import time
import json
import csv
import math
import logging
import threading
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import requests
import numpy as np
from backend.dual_model_engine import dual_model_engine
from backend.news_guard import news_guard

logger = logging.getLogger("layaquant.papertrader")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
STATE_FILE = os.path.join(DATA_DIR, "paper_trading_state.json")
TRADES_CSV = os.path.join(DATA_DIR, "paper_trades.csv")
LIVE_SESSIONS_FILE = os.path.join(DATA_DIR, "live_sessions_history.json")

os.makedirs(DATA_DIR, exist_ok=True)

class PaperTrader:
    def __init__(self, decision_callback=None):
        self.decision_callback = decision_callback
        self.lock = threading.RLock()
        self.worker_thread = None
        self.stop_event = threading.Event()

        # Bot Lifecycle Config
        self.is_running = False
        self.started_at = 0.0
        self.target_duration_hours = 720.0  # Runs continuously until user pauses
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.htf_interval = "1h"
        self.timeline_label = "5m (Scalper)"
        self.session_id = ""
        self.session_start_capital = 10000.0
        self.trades_at_session_start = 0
        self.poll_seconds = 10  # Live real-time action: evaluate market every 10 seconds

        # Institutional Execution & Friction Settings
        self.initial_capital = 10000.0
        self.cash = 10000.0
        self.positions: Dict[str, dict] = {}  # { symbol: position_dict }
        self.max_concurrent_positions: int = 3  # Multi-position capacity (1 to 5 active trades)
        self.confidence_gate = 0.65
        self.allocation_pct = 0.40
        self.use_maker_execution = True
        self.maker_fee_rate = 0.0002   # 0.02% maker post-only limit fill
        self.taker_fee_rate = 0.0005   # 0.05% taker fee (Binance USD-M VIP0)
        self.slippage_bps = 2.0        # 2 bps for limit/stop fills

        # Institutional Microstructure & Execution Upgrades
        self.total_maker_fee_savings_usd = 0.0
        self.partial_tp_ratio = 0.75   # 75% partial take-profit lock (increased from 50% for maximum profit capture)
        self.tp1_atr_mult = 1.4        # TP1 scalp lock at 1.4x ATR (~1:1 R:R)
        self._depth_cache = {}

        # Small-Capital Isolated Margin & Compounding Engine
        self.leverage = 10.0           # 1.0, 3.0, 5.0, 10.0 (10x isolated margin for 10x compounding)
        self.maintenance_margin_rate = 0.005 # 0.5% MMR
        self.auto_compounding = True   # Geometric Quarter-Kelly allocation
        self.milestone_mode = True     # Dynamic Tiered Milestone Compounding (Scales €500 -> €5,000)
        self.consecutive_losses = 0    # Loss streak tracking for defensive throttle
        self.max_equity_risk_pct = 0.045 # Adaptive max account equity risk budget
        self.milestone_targets = [500.0, 1000.0, 1500.0, 2500.0, 3500.0, 5000.0, 10000.0]
        self.last_trade_closed_at = 0.0
        self.trade_cooldown_seconds = 600.0  # 10-min anti-churn cooldown
        self.min_holding_seconds = 300.0     # 5-min minimum holding commitment
        self.max_holding_seconds = 3600.0    # 60-min stale-trade timeout

        # Asymmetric Risk Architecture (Tight 1.35x ATR Initial Stop Loss)
        self.atr_multiplier_stop = 1.35  # Stop loss = 1.35 * ATR14 (cuts losers 40% faster)
        self.target_to_stop_ratio = 2.0  # Take profit = 2.7 * ATR14 (RRR 2.0:1)
        self.trailing_ratchet_atr = 1.2  # Move to breakeven + trail at 1.2 * ATR14 from peak
        self.active_policy_source = "DREAM_ADAPTED"
        self.active_formula = "If RSI < 36.0 or (RSI > 52.0 with Bullish Trend) and Confidence >= 78% --> BUY LONG (SL: 2.2x ATR, TP: 2.8x ATR)"

        # Telemetry State
        self.last_candle_timestamp = 0
        self.last_checked_at = 0.0
        self.current_price = 0.0
        self.current_atr = 0.0
        self.current_funding_rate = 0.0001
        self.current_funding_skew = "neutral"
        self.current_cvd_delta = 0.0
        self.current_htf_regime = "RANGE_BOUND"
        
        self.trades = []
        self.hourly_snapshots = []
        self.audit_log = []
        self._klines_cache = {}
        self._deriv_cache = (0.0, None)

        # Autonomous Multi-Ticker Market Scanner
        self.scanner_mode = True
        self.scanner_universe = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT",
            "XRPUSDT", "BNBUSDT", "AVAXUSDT", "SUIUSDT"
        ]
        self.radar_scan_results = []

        # Strategy selection. TREND is the default: the 5m SCALPER showed negative expectancy in
        # backend/backtest_live_strategy.py; TREND params were chosen on 2020-24 data and held up on 2024-26.
        from backend.trend_strategy import TrendParams
        self.strategy_mode = "TREND"
        self.trend_params = TrendParams(interval="4h", entry_n=120, exit_n=60, stop_atr=4.0,
                                        allow_short=False, risk_pct=0.01)
        self.trend_last_bar: Dict[str, int] = {}
        # Recurring deposit plan (e.g. 100 every 30 days towards a 10,000 target); 0 = off
        self.monthly_deposit = 0.0
        self.last_deposit_ts = 0.0
        self.equity_target = 10000.0
        self.load_state()

        # Resilient auto-recovery: If active positions exist or is_running was True, auto-resume background worker thread
        if self.is_running or (hasattr(self, "positions") and len(self.positions) > 0):
            self.is_running = True
            self.stop_event.clear()
            if self.worker_thread is None or not self.worker_thread.is_alive():
                self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
                self.worker_thread.start()
            logger.info(f"Auto-resumed paper trading background worker thread ({len(self.positions)} active positions).")

    @property
    def position(self):
        if hasattr(self, "positions") and self.positions:
            if self.symbol in self.positions:
                return self.positions[self.symbol]
            return next(iter(self.positions.values()))
        return None

    @position.setter
    def position(self, val):
        if not hasattr(self, "positions"):
            self.positions = {}
        if val is None:
            if self.symbol in self.positions:
                self.positions.pop(self.symbol, None)
            elif self.positions:
                first_k = next(iter(self.positions.keys()))
                self.positions.pop(first_k, None)
        elif isinstance(val, dict) and "symbol" in val:
            self.positions[val["symbol"]] = val

    def load_state(self):
        with self.lock:
            if os.path.exists(STATE_FILE):
                try:
                    with open(STATE_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.is_running = data.get("is_running", False)
                        self.started_at = data.get("started_at", 0.0)
                        self.target_duration_hours = data.get("target_duration_hours", 720.0)
                        self.symbol = data.get("symbol", "BTCUSDT")
                        self.interval = data.get("interval", "5m")
                        self.timeline_label = data.get("timeline_label", "5m (Scalper)")
                        self.session_id = data.get("session_id", "")
                        self.session_start_capital = data.get("session_start_capital", 10000.0)
                        self.trades_at_session_start = data.get("trades_at_session_start", 0)
                        self.initial_capital = data.get("initial_capital", 10000.0)
                        self.cash = data.get("cash", 10000.0)
                        self.max_concurrent_positions = int(data.get("max_concurrent_positions", 3))
                        raw_positions = data.get("positions", {})
                        if isinstance(raw_positions, dict):
                            self.positions = raw_positions
                        elif isinstance(raw_positions, list):
                            self.positions = {p["symbol"]: p for p in raw_positions if isinstance(p, dict) and "symbol" in p}
                        else:
                            self.positions = {}
                        if not self.positions and data.get("position"):
                            p = data["position"]
                            if isinstance(p, dict) and "symbol" in p:
                                self.positions[p["symbol"]] = p
                        self.confidence_gate = max(0.65, data.get("confidence_gate", 0.72))
                        self.allocation_pct = data.get("allocation_pct", 0.40)
                        self.use_maker_execution = data.get("use_maker_execution", True)
                        self.total_maker_fee_savings_usd = float(data.get("total_maker_fee_savings_usd", 0.0))
                        self.partial_tp_ratio = float(data.get("partial_tp_ratio", 0.75))
                        self.tp1_atr_mult = float(data.get("tp1_atr_mult", 1.4))
                        self.max_holding_seconds = float(data.get("max_holding_seconds", 3600.0))
                        self.leverage = min(float(data.get("leverage", 5.0)), 5.0)
                        self.auto_compounding = bool(data.get("auto_compounding", True))
                        self.milestone_mode = bool(data.get("milestone_mode", True))
                        self.consecutive_losses = int(data.get("consecutive_losses", 0))
                        self.max_equity_risk_pct = float(data.get("max_equity_risk_pct", 0.045))
                        self.milestone_targets = data.get("milestone_targets", [500.0, 1000.0, 1500.0, 2500.0, 3500.0, 5000.0, 10000.0])
                        self.atr_multiplier_stop = max(1.0, data.get("sl_atr_mult", 1.35))
                        tp_m = data.get("tp_atr_mult", 2.7)
                        self.target_to_stop_ratio = tp_m / max(0.1, self.atr_multiplier_stop)
                        self.trades = data.get("trades", [])
                        self.hourly_snapshots = data.get("hourly_snapshots", [])
                        self.last_candle_timestamp = data.get("last_candle_timestamp", 0)
                        self.current_price = data.get("current_price", 0.0)
                        self.current_htf_regime = data.get("current_htf_regime", "RANGE_BOUND")
                        self.scanner_mode = bool(data.get("scanner_mode", True))
                        self.scanner_universe = data.get("scanner_universe", [
                            "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT",
                            "XRPUSDT", "BNBUSDT", "AVAXUSDT", "SUIUSDT"
                        ])
                        self.radar_scan_results = data.get("radar_scan_results", [])
                        self.strategy_mode = data.get("strategy_mode", "TREND")
                        self.trend_last_bar = data.get("trend_last_bar", {})
                        self.monthly_deposit = float(data.get("monthly_deposit", 0.0))
                        self.last_deposit_ts = float(data.get("last_deposit_ts", 0.0))
                        self.equity_target = float(data.get("equity_target", 10000.0))
                        self.dual_model_telemetry = data.get("dual_model_telemetry", {})
                        self.active_policy_source = data.get("active_policy_source", "DREAM_ADAPTED")
                        self.active_formula = data.get("active_formula", "If RSI < 36.0 or (RSI > 52.0 with Bullish Trend) and Confidence >= 78% --> BUY LONG (SL: 2.2x ATR, TP: 2.8x ATR)")
                        logger.info(f"Loaded paper trading state from {STATE_FILE}. Active: {self.is_running}")
                except Exception as e:
                    logger.error(f"Error loading state file: {e}")

    def save_state(self):
        with self.lock:
            try:
                data = {
                    "is_running": self.is_running,
                    "started_at": self.started_at,
                    "target_duration_hours": self.target_duration_hours,
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "timeline_label": getattr(self, "timeline_label", "5m (Scalper)"),
                    "session_id": getattr(self, "session_id", ""),
                    "session_start_capital": getattr(self, "session_start_capital", self.initial_capital),
                    "trades_at_session_start": getattr(self, "trades_at_session_start", 0),
                    "initial_capital": self.initial_capital,
                    "cash": self.cash,
                    "max_concurrent_positions": getattr(self, "max_concurrent_positions", 3),
                    "positions": self.positions,
                    "position": self.position,
                    "confidence_gate": self.confidence_gate,
                    "leverage": self.leverage,
                    "auto_compounding": self.auto_compounding,
                    "milestone_mode": getattr(self, "milestone_mode", True),
                    "consecutive_losses": getattr(self, "consecutive_losses", 0),
                    "milestone_targets": getattr(self, "milestone_targets", [500.0, 1000.0, 1500.0, 2500.0, 3500.0, 5000.0, 10000.0]),
                    "max_equity_risk_pct": self.max_equity_risk_pct,
                    "sl_atr_mult": self.atr_multiplier_stop,
                    "tp_atr_mult": round(self.atr_multiplier_stop * self.target_to_stop_ratio, 2),
                    "allocation_pct": self.allocation_pct,
                    "use_maker_execution": self.use_maker_execution,
                    "total_maker_fee_savings_usd": round(getattr(self, "total_maker_fee_savings_usd", 0.0), 2),
                    "partial_tp_ratio": getattr(self, "partial_tp_ratio", 0.75),
                    "tp1_atr_mult": getattr(self, "tp1_atr_mult", 1.4),
                    "max_holding_seconds": getattr(self, "max_holding_seconds", 3600.0),
                    "active_policy_source": getattr(self, "active_policy_source", "DREAM_ADAPTED"),
                    "active_formula": getattr(self, "active_formula", ""),
                    "current_htf_regime": self.current_htf_regime,
                    "trades": self.trades,
                    "hourly_snapshots": self.hourly_snapshots[-300:],
                    "last_candle_timestamp": self.last_candle_timestamp,
                    "last_checked_at": self.last_checked_at,
                    "current_price": self.current_price,
                    "scanner_mode": getattr(self, "scanner_mode", True),
                    "scanner_universe": getattr(self, "scanner_universe", [
                        "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT",
                        "XRPUSDT", "BNBUSDT", "AVAXUSDT", "SUIUSDT"
                    ]),
                    "radar_scan_results": getattr(self, "radar_scan_results", []),
                    "dual_model_telemetry": getattr(self, "dual_model_telemetry", {}),
                    "strategy_mode": getattr(self, "strategy_mode", "TREND"),
                    "trend_last_bar": getattr(self, "trend_last_bar", {}),
                    "monthly_deposit": getattr(self, "monthly_deposit", 0.0),
                    "last_deposit_ts": getattr(self, "last_deposit_ts", 0.0),
                    "equity_target": getattr(self, "equity_target", 10000.0)
                }
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Error writing state file: {e}")

    def apply_dream_policy(self, policy: dict, formula: str = "") -> dict:
        with self.lock:
            if "gate" in policy:
                self.confidence_gate = max(0.50, float(policy["gate"]))
            elif "confidence_gate" in policy:
                self.confidence_gate = max(0.50, float(policy["confidence_gate"]))

            if "sl_atr_mult" in policy:
                self.atr_multiplier_stop = max(0.5, float(policy["sl_atr_mult"]))

            if "tp_atr_mult" in policy:
                tp = float(policy["tp_atr_mult"])
                self.target_to_stop_ratio = max(0.5, tp / max(0.1, self.atr_multiplier_stop))

            self.active_policy_source = "DREAM_ADAPTED"
            if formula:
                self.active_formula = formula
            self.save_state()

        logger.info(f"Deployed Dream-Adapted Policy to Live Paper Trader: Gate={self.confidence_gate*100:.0f}%, SL={self.atr_multiplier_stop}x ATR, TP={self.atr_multiplier_stop*self.target_to_stop_ratio:.2f}x ATR, Formula='{self.active_formula}'")
        return {
            "status": "deployed",
            "active_policy_source": self.active_policy_source,
            "confidence_gate": self.confidence_gate,
            "sl_atr_mult": self.atr_multiplier_stop,
            "tp_atr_mult": round(self.atr_multiplier_stop * self.target_to_stop_ratio, 2),
            "target_to_stop_ratio": round(self.target_to_stop_ratio, 2),
            "active_formula": self.active_formula,
            "message": f"Successfully deployed Dream-Adapted policy to Live Paper Trader! (Gate: {self.confidence_gate*100:.0f}%, SL: {self.atr_multiplier_stop}x, TP: {self.atr_multiplier_stop*self.target_to_stop_ratio:.2f}x)"
        }

    def append_trade_csv(self, trade: dict):
        try:
            file_exists = os.path.exists(TRADES_CSV)
            with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["ID", "Timestamp", "DateTime", "Symbol", "Side", "Price", "Units", "GrossUSD", "FeeUSD", "PnL_USD", "PnL_Pct", "RRR", "Reason"])
                writer.writerow([
                    trade.get("id"),
                    trade.get("timestamp"),
                    trade.get("datetime"),
                    trade.get("symbol"),
                    trade.get("side"),
                    trade.get("price"),
                    trade.get("units"),
                    trade.get("gross_usd"),
                    trade.get("fee_usd"),
                    trade.get("pnl_usd", 0.0),
                    trade.get("pnl_pct", 0.0),
                    trade.get("rrr", "2.5:1"),
                    trade.get("reason", "")
                ])
        except Exception as e:
            logger.error(f"Error writing trade CSV: {e}")

    def start(self, symbol="BTCUSDT", interval="5m", initial_capital=10000.0, confidence_gate=0.72, target_duration_hours=720.0):
        with self.lock:
            if self.is_running:
                return {"status": "already_running", "message": "Bot is already running."}
            
            self.symbol = symbol.upper()
            self.interval = (interval or "5m").lower()

            # Multi-timeline calibration
            if self.interval == "1m":
                self.htf_interval = "15m"
                self.poll_seconds = 5
                self.timeline_label = "1m (Ultra-Scalp)"
            elif self.interval == "5m":
                self.htf_interval = "1h"
                self.poll_seconds = 12
                self.timeline_label = "5m (Scalper)"
            elif self.interval == "15m":
                self.htf_interval = "4h"
                self.poll_seconds = 25
                self.timeline_label = "15m (Day Trader)"
            elif self.interval == "1h":
                self.htf_interval = "1d"
                self.poll_seconds = 60
                self.timeline_label = "1h (Swing Trader)"
            elif self.interval == "4h":
                self.htf_interval = "1d"
                self.poll_seconds = 120
                self.timeline_label = "4h (Position)"
            else:  # 1d
                self.htf_interval = "1w"
                self.poll_seconds = 300
                self.timeline_label = "1d (Macro Trend)"

            self.confidence_gate = max(0.65, confidence_gate)
            self.target_duration_hours = target_duration_hours or 720.0
            
            if initial_capital and initial_capital > 0:
                self.initial_capital = float(initial_capital)
                if self.position is None:
                    self.cash = float(initial_capital)

            now_t = time.time()
            self.session_id = f"live_{self.symbol.lower()}_{self.interval}_{int(now_t)}"
            self.started_at = now_t
            self.session_start_capital = self.cash
            self.trades_at_session_start = len(self.trades)

            self.is_running = True
            self.stop_event.clear()
            self.save_state()

        logger.info(f"Started Persistent Live Action Recording for {self.symbol} on {self.timeline_label}. Target: {self.target_duration_hours}h.")
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

        return {
            "status": "started",
            "session_id": self.session_id,
            "symbol": self.symbol,
            "interval": self.interval,
            "timeline_label": self.timeline_label,
            "capital": self.cash,
            "started_at": self.started_at,
            "message": f"Started continuous live recording on {self.timeline_label}. Runs in the background until paused."
        }

    def stop(self) -> dict:
        with self.lock:
            if not self.is_running:
                return {"status": "not_running", "message": "Bot is not currently active."}

            self.is_running = False
            self.stop_event.set()

            now_t = time.time()
            elapsed_sec = (now_t - self.started_at) if self.started_at > 0 else 0
            elapsed_h = int(elapsed_sec // 3600)
            elapsed_m = int((elapsed_sec % 3600) // 60)
            elapsed_s = int(elapsed_sec % 60)

            fee_rate = self.maker_fee_rate if self.use_maker_execution else self.taker_fee_rate
            pos_val = 0.0
            if self.position:
                pos_side = self.position.get("side", "LONG")
                entry = self.position["entry_price"]
                units = self.position["units"]
                margin = self.position.get("margin_allocated", (units * entry) / self.position.get("leverage", self.leverage))
                pos_px = float(self.position.get("current_price", entry))
                if pos_side == "LONG":
                    gross_pnl = (pos_px - entry) * units
                else:
                    gross_pnl = (entry - pos_px) * units
                est_fee = (units * pos_px) * fee_rate
                pos_val = max(0.0, margin + gross_pnl - est_fee)

            final_equity = self.cash + pos_val
            start_cap = getattr(self, "session_start_capital", self.initial_capital)
            net_pnl = final_equity - start_cap
            net_pnl_pct = (net_pnl / max(start_cap, 1.0)) * 100.0

            start_idx = getattr(self, "trades_at_session_start", 0)
            session_trades = self.trades[start_idx:]
            closed_session_trades = [t for t in session_trades if any(k in t.get("side", "") for k in ["SELL", "CLOSE", "STOP_LOSS", "TAKE_PROFIT", "LIQUIDATION"])]
            winning_trades = [t for t in closed_session_trades if t.get("pnl_usd", 0) > 0]
            win_rate = (len(winning_trades) / max(1, len(closed_session_trades)) * 100.0) if closed_session_trades else 0.0

            gross_win = sum(t.get("pnl_usd", 0) for t in winning_trades)
            gross_loss = abs(sum(t.get("pnl_usd", 0) for t in closed_session_trades if t.get("pnl_usd", 0) < 0))
            profit_factor = (gross_win / max(1.0, gross_loss)) if gross_loss > 0 else (2.0 if gross_win > 0 else 1.0)

            from datetime import datetime
            session_record = {
                "session_id": getattr(self, "session_id", f"live_{int(now_t)}"),
                "symbol": self.symbol,
                "interval": self.interval,
                "timeline_label": getattr(self, "timeline_label", f"{self.interval} Trader"),
                "started_at": self.started_at,
                "started_datetime": datetime.fromtimestamp(self.started_at).strftime("%Y-%m-%d %H:%M:%S") if self.started_at > 0 else "",
                "paused_at": now_t,
                "paused_datetime": datetime.fromtimestamp(now_t).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_formatted": f"{elapsed_h}h {elapsed_m}m {elapsed_s}s",
                "duration_seconds": elapsed_sec,
                "starting_capital": round(start_cap, 2),
                "ending_equity": round(final_equity, 2),
                "net_pnl": round(net_pnl, 2),
                "net_pnl_pct": round(net_pnl_pct, 2),
                "total_trades": len(session_trades),
                "closed_trades": len(closed_session_trades),
                "winning_trades": len(winning_trades),
                "win_rate": round(win_rate, 1),
                "profit_factor": round(profit_factor, 2),
                "open_position": self.position,
                "trades": session_trades
            }

            self._save_session_record(session_record)
            self.save_state()

        logger.info(f"Paused Live Paper Trading Session {session_record['session_id']}. Duration: {session_record['duration_formatted']}, PnL: ${net_pnl:.2f}")
        return {"status": "paused", "session": session_record}

    def _save_session_record(self, record: dict):
        try:
            sessions = []
            if os.path.exists(LIVE_SESSIONS_FILE):
                with open(LIVE_SESSIONS_FILE, "r", encoding="utf-8") as f:
                    sessions = json.load(f)
            sessions.append(record)
            with open(LIVE_SESSIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving live session record: {e}")

    def get_live_sessions(self) -> List[dict]:
        if os.path.exists(LIVE_SESSIONS_FILE):
            try:
                with open(LIVE_SESSIONS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def reset(self, initial_capital=10000.0):
        with self.lock:
            self.is_running = False
            self.stop_event.set()
            self.initial_capital = initial_capital
            self.cash = initial_capital
            self.session_start_capital = initial_capital
            self.consecutive_losses = 0
            self.positions = {}
            self.position = None
            self.total_maker_fee_savings_usd = 0.0
            self.trades = []
            self.hourly_snapshots = []
            self.audit_log = []
            self.started_at = time.time()
            self.last_candle_timestamp = 0
            self.save_state()
        if os.path.exists(TRADES_CSV):
            try:
                os.remove(TRADES_CSV)
            except Exception:
                pass
        return {"status": "reset", "initial_capital": initial_capital}

    def deposit_cash(self, amount: float):
        with self.lock:
            if amount <= 0:
                return {"status": "error", "message": "Deposit amount must be positive."}
            self.cash += amount
            self.initial_capital += amount
            self.save_state()
            return {"status": "success", "deposited": amount, "new_cash": self.cash, "initial_capital": self.initial_capital}

    DEPOSIT_INTERVAL_S = 30 * 86400

    def set_recurring_deposit(self, amount: float, target: Optional[float] = None) -> dict:
        """Add `amount` every 30 days (first one 30 days from now); 0 turns it off."""
        with self.lock:
            if amount < 0:
                return {"status": "error", "message": "Deposit amount cannot be negative."}
            self.monthly_deposit = float(amount)
            self.last_deposit_ts = time.time()
            if target:
                self.equity_target = float(target)
            self.save_state()
            return {"status": "success", "monthly_deposit": self.monthly_deposit, "equity_target": self.equity_target,
                    "next_deposit_at": self.last_deposit_ts + self.DEPOSIT_INTERVAL_S}

    def _apply_recurring_deposit(self):
        """Credit every scheduled deposit that has come due (catches up after downtime)."""
        amount = getattr(self, "monthly_deposit", 0.0)
        if amount > 0 and not self.last_deposit_ts:
            self.last_deposit_ts = time.time()
        while amount > 0 and time.time() - self.last_deposit_ts >= self.DEPOSIT_INTERVAL_S:
            self.last_deposit_ts += self.DEPOSIT_INTERVAL_S
            self.deposit_cash(amount)
            logger.info(f"[DEPOSIT] recurring deposit of {amount:.2f} credited; total deposited {self.initial_capital:.2f}")

    def set_cash(self, cash: float):
        with self.lock:
            if cash < 0:
                return {"status": "error", "message": "Cash amount cannot be negative."}
            self.cash = float(cash)
            self.initial_capital = float(cash)
            self.session_start_capital = float(cash)
            self.consecutive_losses = 0
            self.save_state()
            return {"status": "success", "new_cash": self.cash, "initial_capital": self.initial_capital}

    def set_budget(self, allocation_pct: float):
        with self.lock:
            self.allocation_pct = max(0.05, min(1.0, allocation_pct))
            self.save_state()
            return {"status": "success", "allocation_pct": self.allocation_pct}

    def fetch_klines(self, interval="1h", limit=50, symbol: str = None) -> List[dict]:
        """Fetch candles with taker buy volume for CVD derivation for any symbol."""
        now = time.time()
        target_sym = (symbol or self.symbol).upper()
        cache_key = (target_sym, interval, limit)
        if cache_key in self._klines_cache and (now - self._klines_cache[cache_key][0]) < 3.0:
            return self._klines_cache[cache_key][1]

        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": target_sym, "interval": interval, "limit": limit}
        try:
            resp = requests.get(url, params=params, timeout=4.0)
            if resp.status_code == 200:
                raw = resp.json()
                candles = []
                for k in raw:
                    tot_vol = float(k[5])
                    taker_buy = float(k[9]) if len(k) > 9 else tot_vol * 0.5
                    candles.append({
                        "timestamp": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": tot_vol,
                        "taker_buy": taker_buy,
                        "taker_sell": max(0.0, tot_vol - taker_buy),
                        "cvd_delta": taker_buy - (tot_vol - taker_buy)
                    })
                self._klines_cache[cache_key] = (now, candles)
                return candles
        except Exception as e:
            logger.warning(f"Binance kline fetch ({target_sym} {interval}) error: {e}")

        # Isolated Fallback: Check if we have an older cached candle for this specific target_sym
        for (c_sym, c_int, _), (_, c_candles) in self._klines_cache.items():
            if c_sym == target_sym and c_int == interval and c_candles:
                return c_candles

        # Check if symbol is in active positions and use its own known price
        if hasattr(self, "positions") and target_sym in self.positions:
            pos = self.positions[target_sym]
            p = float(pos.get("current_price", pos.get("entry_price", 0.0)))
            if p > 0:
                now_ms = int(time.time() * 1000)
                return [{
                    "timestamp": now_ms, "open": p, "high": p * 1.0005, "low": p * 0.9995, "close": p,
                    "volume": 10.0, "taker_buy": 5.0, "taker_sell": 5.0, "cvd_delta": 0.0
                }]

        # Do NOT invent prices from other coins or self.current_price!
        return []

    def fetch_derivatives_data(self, symbol: str = None) -> dict:
        """Fetch Binance perpetual funding rate and mark/index basis for any symbol."""
        now = time.time()
        target_sym = (symbol or self.symbol).upper()
        if self._deriv_cache[1] is not None and (now - self._deriv_cache[0]) < 3.0 and target_sym == self.symbol:
            return self._deriv_cache[1]

        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={target_sym}"
        try:
            resp = requests.get(url, timeout=3.0)
            if resp.status_code == 200:
                d = resp.json()
                fr = float(d.get("lastFundingRate", 0.0001))
                mark = float(d.get("markPrice", 0.0))
                idx = float(d.get("indexPrice", 0.0))
                skew = "neutral"
                if fr > 0.0003:
                    skew = "crowded_long"
                elif fr < -0.0003:
                    skew = "crowded_short"
                data = {"funding_rate": fr, "funding_skew": skew, "mark_price": mark, "index_price": idx}
                if target_sym == self.symbol:
                    self._deriv_cache = (now, data)
                return data
        except Exception as e:
            logger.warning(f"Derivatives funding rate fetch error ({target_sym}): {e}")
        return {"funding_rate": 0.0001, "funding_skew": "neutral", "mark_price": self.current_price, "index_price": self.current_price}

    def _compute_atr_and_indicators(self, candles: List[dict]):
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]
        cvd_deltas = [c.get("cvd_delta", 0.0) for c in candles]

        # 1. Average True Range (ATR 14)
        tr_list = []
        for i in range(1, len(candles)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr_list.append(max(hl, hc, lc))
        
        atr_14 = float(np.mean(tr_list[-14:])) if len(tr_list) >= 14 else (closes[-1] * 0.015)
        atr_avg28 = float(np.mean(tr_list[-28:])) if len(tr_list) >= 28 else atr_14
        atr_expansion = float(atr_14 / max(atr_avg28, 1e-4))

        # 2. Moving averages
        sma_fast = float(np.mean(closes[-10:])) if len(closes) >= 10 else closes[-1]
        sma_slow = float(np.mean(closes[-30:])) if len(closes) >= 30 else closes[-1]

        # 3. RSI 14
        if len(closes) >= 15:
            diffs = np.diff(closes[-15:])
            gains = diffs[diffs > 0].sum() / 14.0
            losses = -diffs[diffs < 0].sum() / 14.0
            rs = (gains / max(losses, 1e-6))
            rsi = float(100.0 - (100.0 / (1.0 + rs)))
        else:
            rsi = 50.0

        # 4. Volume Spike & CVD momentum
        avg_vol = float(np.mean(volumes[-10:])) if len(volumes) >= 10 else volumes[-1]
        vol_spike = float(volumes[-1] / max(avg_vol, 1e-4))
        cvd_momentum = float(np.sum(cvd_deltas[-3:]))

        return atr_14, atr_expansion, sma_fast, sma_slow, rsi, vol_spike, cvd_momentum

    def classify_htf_macro_regime(self, symbol: str = None) -> str:
        """Evaluate 4h higher timeframe to classify macro market regime for any symbol."""
        htf_candles = self.fetch_klines(interval=self.htf_interval, limit=35, symbol=symbol)
        if not htf_candles or len(htf_candles) < 15:
            return "RANGE_BOUND"

        atr_14, atr_exp, sma_fast, sma_slow, rsi, vol_spike, _ = self._compute_atr_and_indicators(htf_candles)
        last_price = htf_candles[-1]["close"]
        spread_pct = ((sma_fast - sma_slow) / max(sma_slow, 1e-6)) * 100.0

        if atr_exp > 1.6 and abs(spread_pct) < 0.4:
            return "TOXIC_CHOP"
        elif spread_pct > 0.5 and last_price >= sma_fast:
            return "TREND_BULL"
        elif spread_pct < -0.5 and last_price <= sma_fast:
            return "TREND_BEAR"
        else:
            return "RANGE_BOUND"

    def fetch_order_book_depth(self, symbol: str, limit: int = 20) -> dict:
        """
        Fetch Level 2 Order Book Depth from Binance.
        Computes resting Bid Depth ($), Ask Depth ($), and Depth Ratio (Bids / Asks).
        """
        now = time.time()
        if not hasattr(self, "_depth_cache"):
            self._depth_cache = {}
        if symbol in self._depth_cache:
            entry = self._depth_cache[symbol]
            if now - entry["ts"] < 3.0:
                return entry["data"]

        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                bid_depth_usd = sum(float(b[0]) * float(b[1]) for b in bids)
                ask_depth_usd = sum(float(a[0]) * float(a[1]) for a in asks)
                depth_ratio = bid_depth_usd / max(1.0, ask_depth_usd)
                best_bid = float(bids[0][0]) if bids else 0.0
                best_ask = float(asks[0][0]) if asks else 0.0
                spread_bps = (((best_ask - best_bid) / max(1e-6, best_bid)) * 10000.0) if best_bid > 0 else 0.0

                res = {
                    "bid_depth_usd": round(bid_depth_usd, 2),
                    "ask_depth_usd": round(ask_depth_usd, 2),
                    "depth_ratio": round(depth_ratio, 2),
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread_bps": round(spread_bps, 2)
                }
                self._depth_cache[symbol] = {"ts": now, "data": res}
                return res
        except Exception as e:
            return {
                "bid_depth_usd": 100000.0,
                "ask_depth_usd": 100000.0,
                "depth_ratio": 1.0,
                "best_bid": 0.0,
                "best_ask": 0.0,
                "spread_bps": 1.5
            }

    def scan_market_universe(self) -> List[dict]:
        """
        Autonomous Multi-Ticker Market Radar:
        Scans all perpetual tickers in self.scanner_universe in real time.
        Computes 4H Trend, 5m ATR, Bollinger-Keltner Squeeze, CVD Delta, and Confluence Score (0 to 100).
        Returns list of scored symbols sorted by score descending.
        """
        radar_results = []
        try:
            from backend.news_engine import news_engine
            news_state = news_engine.get_aggregate_market_sentiment()
            base_news_score = float(news_state.get("score", 0.0))
            fng_data = news_engine.get_fear_and_greed()
        except Exception:
            base_news_score = 0.0
            fng_data = {"value": 50, "value_classification": "Neutral"}

        for sym in self.scanner_universe:
            try:
                candles_5m = self.fetch_klines(interval="5m", limit=30, symbol=sym)
                if not candles_5m or len(candles_5m) < 15:
                    continue

                atr_14, atr_exp, sma_fast, sma_slow, rsi, vol_spike, cvd_mom = self._compute_atr_and_indicators(candles_5m)
                cur_px = float(candles_5m[-1]["close"])
                open_px = float(candles_5m[-5]["close"]) if len(candles_5m) >= 5 else float(candles_5m[0]["close"])
                chg_5m = ((cur_px - open_px) / max(open_px, 1e-4)) * 100.0

                # 4H Higher Timeframe Regime
                regime = self.classify_htf_macro_regime(symbol=sym)

                # Bollinger-Keltner Volatility Squeeze
                closes = [c["close"] for c in candles_5m]
                sma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else cur_px
                std20 = float(np.std(closes[-20:])) if len(closes) >= 20 else (cur_px * 0.01)
                bb_upper = sma20 + (2.0 * std20)
                bb_lower = sma20 - (2.0 * std20)
                kc_upper = sma20 + (1.5 * atr_14)
                kc_lower = sma20 - (1.5 * atr_14)
                is_squeeze = bool(bb_upper < kc_upper and bb_lower > kc_lower)

                # Confluence Scoring Engine (0 - 100)
                spread_pct = ((sma_fast - sma_slow) / max(sma_slow, 1e-6)) * 100.0

                # Ticker-Specific News, Twitter/X Pulse & Whale Alert Intelligence
                ticker_sent = news_engine.get_ticker_sentiment(sym)

                # Microstructure: Fetch Level 2 Order Book Depth Imbalance
                depth_info = self.fetch_order_book_depth(sym)

                # Dual-Model Consensus Evaluation (Model A Macro Governor + Model B Micro Sniper)
                dual_res = dual_model_engine.evaluate_opportunity(
                    symbol=sym,
                    htf_regime=regime,
                    candles_5m=candles_5m,
                    atr_14=atr_14,
                    cvd_delta=cvd_mom,
                    rsi=rsi,
                    spread_pct=spread_pct,
                    news_score=ticker_sent.get("score", base_news_score),
                    atr_expansion=atr_exp,
                    ticker_sentiment=ticker_sent,
                    depth_ratio=depth_info["depth_ratio"],
                    spread_bps=depth_info["spread_bps"],
                    fear_and_greed=fng_data
                )

                # Active position check across multi-position portfolio
                is_active = (sym in getattr(self, "positions", {})) or (self.position is not None and self.position.get("symbol") == sym)

                status_label = "ACTIVE 10x TRADE" if is_active else (
                    "DUAL CONSENSUS" if dual_res["dual_consensus_achieved"] else (
                        "GOV VETO" if "VETO" in dual_res["consensus_status"] else "SCANNING"
                    )
                )

                radar_results.append({
                    "symbol": sym,
                    "price": round(cur_px, 4 if cur_px < 1.0 else 2),
                    "change_5m": round(chg_5m, 2),
                    "htf_regime": regime,
                    "is_squeeze": is_squeeze,
                    "squeeze_status": "SQUEEZE_ON" if is_squeeze else "Normal",
                    "cvd_mom": round(cvd_mom, 1),
                    "rsi": round(rsi, 1),
                    "atr_14": round(atr_14, 4 if atr_14 < 1.0 else 2),
                    "price_raw": cur_px,
                    "atr_raw": atr_14,
                    "score": round(float(dual_res["consensus_score"]), 1),
                    "choice": dual_res["final_action"],
                    "dual_consensus_achieved": dual_res["dual_consensus_achieved"],
                    "consensus_status": dual_res["consensus_status"],
                    "synthesis_reasoning": dual_res["synthesis_reasoning"],
                    "recommended_leverage": dual_res["recommended_leverage"],
                    "news_score": ticker_sent.get("score", 0.0),
                    "news_label": ticker_sent.get("label", "Neutral"),
                    "social_buzz": ticker_sent.get("social_buzz", 50.0),
                    "top_catalyst": ticker_sent.get("top_catalyst", ""),
                    "top_source": ticker_sent.get("top_source", ""),
                    "ticker_sentiment": ticker_sent,
                    "model_a_macro": dual_res["model_a_macro"],
                    "model_b_micro": dual_res["model_b_micro"],
                    "depth_ratio": depth_info["depth_ratio"],
                    "spread_bps": depth_info["spread_bps"],
                    "vpin_toxicity": dual_res.get("vpin_toxicity", "CLEAN"),
                    "bid_depth_usd": depth_info["bid_depth_usd"],
                    "ask_depth_usd": depth_info["ask_depth_usd"],
                    "is_active_trade": is_active,
                    "status_label": status_label
                })
            except Exception as e:
                logger.debug(f"Error scanning {sym}: {e}")

        # Sort descending by score
        radar_results.sort(key=lambda x: (x["is_active_trade"], x["score"]), reverse=True)
        with self.lock:
            self.radar_scan_results = radar_results
            self.last_radar_scan_time = time.time()
            if radar_results:
                active_items = [r for r in radar_results if r["is_active_trade"]]
                top_item = active_items[0] if active_items else radar_results[0]
                self.dual_model_telemetry = {
                    "symbol": top_item["symbol"],
                    "price": top_item["price"],
                    "final_action": top_item["choice"],
                    "consensus_status": top_item.get("consensus_status", "SCANNING"),
                    "consensus_score": top_item.get("score", 0.0),
                    "dual_consensus_achieved": top_item.get("dual_consensus_achieved", False),
                    "synthesis_reasoning": top_item.get("synthesis_reasoning", ""),
                    "model_a_macro": top_item.get("model_a_macro", {}),
                    "model_b_micro": top_item.get("model_b_micro", {}),
                    "ticker_sentiment": top_item.get("ticker_sentiment", {}),
                    "recommended_leverage": top_item.get("recommended_leverage", 10.0),
                    "timestamp": time.time()
                }
            self.last_radar_scan_time = time.time()
        return radar_results

    def compute_compounding_allocation(self, equity: float, stop_dist_pct: float) -> tuple[float, float, float]:
        """
        Calculates optimal compounding position size using Dynamic Milestone Tiered Kelly
        specifically engineered to scale €500 -> €5,000 without crossing the Kelly overbetting cliff:
        - Phase 1 (< €1,500): Scrappy Growth (4.0% - 8.5% equity risk cap)
        - Phase 2 (€1,500 - €3,500): Growth Expansion (3.0% - 6.5% equity risk cap)
        - Phase 3 (€3,500 - €5,000+): Milestone Capital Lock (2.0% - 4.5% equity risk cap)
        - Consecutive Loss Throttle: 40% risk reduction after loss
        """
        # Realized exit legs only (entry rows like SELL_SHORT carry pnl 0 and must not count as losses)
        exit_legs = [t for t in self.trades
                     if any(k in t.get("side", "") for k in ["CLOSE", "PARTIAL_TP", "SCRATCH"])
                     and not t.get("side", "").startswith("MANUAL")]
        wins = [t["pnl_usd"] for t in exit_legs if t.get("pnl_usd", 0) > 0]
        losses = [-t["pnl_usd"] for t in exit_legs if t.get("pnl_usd", 0) < 0]

        # Kelly from REALIZED stats. Priors are deliberately unexciting until there is evidence:
        # the backtest (backend/backtest_live_strategy.py) shows no positive edge for this signal.
        if len(exit_legs) >= 30 and wins and losses:
            p = len(wins) / (len(wins) + len(losses))
            b = float(np.mean(wins)) / max(1e-9, float(np.mean(losses)))
        else:
            p, b = 0.50, 1.0
        kelly_full = (p * (b + 1.0) - 1.0) / b if b > 0 else -1.0

        probe_risk = 0.005  # 0.5% probe size when there is no demonstrated edge
        if kelly_full <= 0:
            risk_pct = probe_risk
        elif getattr(self, "milestone_mode", True) or equity <= 3000.0:
            # Dynamic Milestone Tiered Kelly: the phase numbers are CAPS, never floors
            if equity < 1500.0:
                risk_pct = min(kelly_full * 0.45, 0.085)
            elif equity < 3500.0:
                risk_pct = min(kelly_full * 0.35, 0.065)
            else:
                risk_pct = min(kelly_full * 0.25, 0.045)
            risk_pct = max(probe_risk, risk_pct)

            # Throttle risk by 40% if recovering from a recent loss
            if getattr(self, "consecutive_losses", 0) >= 1:
                risk_pct = max(probe_risk, risk_pct * 0.60)
        else:
            risk_pct = min(max(probe_risk, kelly_full * 0.25), self.max_equity_risk_pct)

        risk_capital = equity * risk_pct
        safe_stop_pct = max(float(stop_dist_pct), 0.005)
        desired_notional = risk_capital / safe_stop_pct
        
        required_margin = desired_notional / max(1.0, self.leverage)
        
        # Maintain 20% liquid cash buffer to guard against margin calls
        allocated_margin = min(required_margin, self.cash * 0.80)
        allocated_margin = max(15.0, allocated_margin) # minimum $15 test margin
        
        if not self.auto_compounding:
            allocated_margin = min(self.cash * self.allocation_pct, self.cash * 0.85)
            desired_notional = allocated_margin * self.leverage

        return float(allocated_margin), float(desired_notional), float(risk_pct)

    def _manage_scalper_positions(self):
        """Exit management for positions opened by the 5m scalper (stops, TP1/runner, liquidation, timeout)."""
        fee_rate = self.maker_fee_rate if self.use_maker_execution else self.taker_fee_rate
        # Stops, timeouts and forced exits are market orders regardless of entry style
        exit_fee_rate = self.taker_fee_rate
        exit_slip = self.slippage_bps / 10000.0

        for pos_sym, pos in list(self.positions.items()):
            if pos.get("strategy") == "TREND":
                continue
            pos_candles = self.fetch_klines(interval=self.interval, limit=10, symbol=pos_sym)
            if not pos_candles:
                continue
            pos_latest_candle = pos_candles[-1]
            side = pos.get("side", "LONG").upper()
            entry_px = pos["entry_price"]
            units = pos["units"]
            margin_alloc = pos.get("margin_allocated", (units * entry_px) / max(1.0, pos.get("leverage", self.leverage)))
            pos_lev = pos.get("leverage", self.leverage)
            liq_px = pos.get("liquidation_price", 0.0)

            pos["current_price"] = pos_latest_candle["close"]
            pos_px = float(pos_latest_candle["close"])

            now_ms = int(time.time() * 1000)
            last_eval_ms = int(pos.get("last_eval_ms", pos.get("entry_time", time.time()) * 1000))
            obs_high = obs_low = pos_px
            for c in pos_candles:
                if c["timestamp"] >= last_eval_ms:
                    obs_high = max(obs_high, c["high"])
                    obs_low = min(obs_low, c["low"])
            pos["last_eval_ms"] = now_ms
            pos_latest_candle = {**pos_latest_candle, "high": obs_high, "low": obs_low}

            # Anomaly Guard: If price diverges by >25% from entry price on a 5m bar, it is a corrupted tick
            if abs(pos_px - entry_px) / max(0.0001, entry_px) > 0.25:
                logger.error(f"[PRICE ANOMALY BLOCKED] Corrupted tick for {pos_sym}: Entry=${entry_px}, Tick=${pos_px}. Skipping evaluation.")
                continue

            if side == "LONG":
                # Ratchet from the peak of EARLIER observations; this window's high applies next poll,
                # otherwise a stop raised by this window's high gets "hit" by this window's (earlier) low.
                prev_peak = pos.get("highest_price", entry_px)
                pos["highest_price"] = max(prev_peak, pos_latest_candle["high"])

                # A. Check Trailing Ratchet Activation (Triggered at +1.5x ATR14)
                breakeven_threshold = pos.get("breakeven_trigger", entry_px + (1.5 * pos["atr_14"]))
                if prev_peak >= breakeven_threshold:
                    if not pos.get("is_trailing", False):
                        pos["is_trailing"] = True
                        pos["stop_loss_price"] = max(pos["stop_loss_price"], entry_px * 1.002)
                        px_lbl = f"${pos['stop_loss_price']:.4f}" if pos['stop_loss_price'] < 1.0 else f"${pos['stop_loss_price']:.2f}"
                        logger.info(f"[RISK RATCHET] Breakeven locked for LONG {pos_sym} at {px_lbl}")

                    ratchet_stop = prev_peak - (1.5 * pos["atr_14"])
                    if ratchet_stop > pos["stop_loss_price"]:
                        pos["stop_loss_price"] = ratchet_stop

                # B. Liquidation Check (Isolated Margin)
                if liq_px > 0 and pos_latest_candle["low"] <= liq_px:
                    exit_price = liq_px
                    pnl = -margin_alloc
                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": "LIQUIDATION_CLOSE_LONG",
                        "price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "units": round(units, 6),
                        "gross_usd": round(margin_alloc, 2),
                        "fee_usd": 0.0,
                        "pnl_usd": round(pnl, 2),
                        "pnl_pct": -100.0,
                        "rrr": "0:1",
                        "reason": f"Isolated Margin Liquidation @ ${exit_price:.4f} [{pos_lev:.0f}x]" if exit_price < 1.0 else f"Isolated Margin Liquidation @ ${exit_price:.2f} [{pos_lev:.0f}x]"
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    self.consecutive_losses += 2
                    logger.warning(f"[LIQUIDATION] LONG on {pos_sym} liquidated @ ${exit_price:.4f} | Loss: -${margin_alloc:.2f}")
                    self.last_trade_closed_at = time.time()
                    self.positions.pop(pos_sym, None)
                    continue

                # B1. Two-Stage Partial Take-Profit (50% Scalp Lock at 1.2x ATR)
                tp1_threshold = pos.get("tp1_price", entry_px + (getattr(self, "tp1_atr_mult", 1.2) * pos["atr_14"]))
                stop_hit = pos_latest_candle["low"] <= pos["stop_loss_price"]
                if not stop_hit and not pos.get("tp1_hit", False) and pos_latest_candle["high"] >= tp1_threshold:
                    ratio = getattr(self, "partial_tp_ratio", 0.50)
                    close_units = pos["units"] * ratio
                    close_margin = margin_alloc * ratio
                    exit_price = tp1_threshold
                    gross_pnl = (exit_price - entry_px) * close_units
                    fee = (close_units * exit_price) * fee_rate
                    net_pnl = gross_pnl - fee

                    pos["units"] -= close_units
                    pos["margin_allocated"] -= close_margin
                    pos["tp1_hit"] = True
                    margin_alloc = pos["margin_allocated"]
                    units = pos["units"]
                    self.cash += max(0.0, close_margin + net_pnl)
                    pnl_pct = (net_pnl / max(close_margin, 1.0)) * 100.0

                    # Instantly de-risk remaining 50% runner to guaranteed breakeven (+0.2x ATR above entry)
                    pos["stop_loss_price"] = max(pos["stop_loss_price"], entry_px + (0.2 * pos["atr_14"]))
                    pos["is_trailing"] = True

                    px_str = f"${exit_price:.4f}" if exit_price < 1.0 else f"${exit_price:.2f}"
                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": "PARTIAL_TP1_LONG",
                        "price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "entry_price": round(entry_px, 4 if entry_px < 1.0 else 2),
                        "exit_price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "units": round(close_units, 6),
                        "gross_usd": round(close_units * exit_price, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "rrr": "1:1",
                        "reason": f"TP1 Scalp Lock: {int(ratio * 100)}% closed at {px_str} (+${net_pnl:.2f}, +{pnl_pct:.1f}%). Free runner stop ratcheted to entry."
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    logger.info(f"[TP1 SCALP LOCK] LONG {pos_sym} 50% locked @ {px_str} (+${net_pnl:.2f}) | Risk-Free Runner Active")

                # C. Take-Profit Trigger
                elif not stop_hit and pos_latest_candle["high"] >= pos["take_profit_price"]:
                    exit_price = pos["take_profit_price"]
                    gross_pnl = (exit_price - entry_px) * units
                    fee = (units * exit_price) * fee_rate
                    net_pnl = gross_pnl - fee
                    self.cash += max(0.0, margin_alloc + net_pnl)
                    pnl_pct = (net_pnl / max(margin_alloc, 1.0)) * 100.0
                    self.consecutive_losses = 0

                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": "TAKE_PROFIT_CLOSE_LONG",
                        "price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "entry_price": round(entry_px, 4 if entry_px < 1.0 else 2),
                        "exit_price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "units": round(units, 6),
                        "gross_usd": round(units * exit_price, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "rrr": f"{self.target_to_stop_ratio:.1f}:1",
                        "reason": f"Asymmetric Target Hit (+${net_pnl:.2f}, +{pnl_pct:.1f}%) [{pos_lev:.0f}x Leverage]"
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    px_str = f"${exit_price:.4f}" if exit_price < 1.0 else f"${exit_price:.2f}"
                    logger.info(f"[ASYMMETRIC WIN] TP Hit on LONG {pos_sym} @ {px_str} | PnL: +${net_pnl:.2f}")
                    self.last_trade_closed_at = time.time()
                    self.positions.pop(pos_sym, None)
                    continue

                # D. Stop-Loss Trigger
                elif pos_latest_candle["low"] <= pos["stop_loss_price"]:
                    exit_price = min(pos["stop_loss_price"], pos_px) * (1.0 - exit_slip)
                    gross_pnl = (exit_price - entry_px) * units
                    fee = (units * exit_price) * exit_fee_rate
                    net_pnl = gross_pnl - fee
                    self.cash += max(0.0, margin_alloc + net_pnl)
                    pnl_pct = (net_pnl / max(margin_alloc, 1.0)) * 100.0
                    if net_pnl > 0:
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1

                    is_trail = pos.get("is_trailing", False)
                    reason_tag = "Trailing Ratchet Stop Hit" if is_trail else f"Dynamic {self.atr_multiplier_stop:.1f}x ATR Stop Loss"
                    px_str = f"${exit_price:.4f}" if exit_price < 1.0 else f"${exit_price:.2f}"

                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": "STOP_LOSS_CLOSE_LONG",
                        "price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "entry_price": round(entry_px, 4 if entry_px < 1.0 else 2),
                        "exit_price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "units": round(units, 6),
                        "gross_usd": round(units * exit_price, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "rrr": "2.5:1",
                        "reason": f"{reason_tag} ({px_str}) [{pos_lev:.0f}x Leverage]"
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    logger.info(f"[RISK EXIT] {reason_tag} on LONG {pos_sym} @ {px_str} | PnL: ${net_pnl:.2f}")
                    self.last_trade_closed_at = time.time()
                    self.positions.pop(pos_sym, None)
                    continue

                # E. Dynamic Momentum Scratch (Long)
                holding_seconds = time.time() - pos.get("entry_time", time.time())
                max_hold = getattr(self, "max_holding_seconds", 3600.0)
                should_scratch = False
                scratch_tag = ""
                px_str = f"${pos_px:.4f}" if pos_px < 1.0 else f"${pos_px:.2f}"

                if not pos.get("tp1_hit", False) and holding_seconds >= max_hold:
                    should_scratch = True
                    scratch_tag = f"60m Max Holding Scratch ({holding_seconds/60:.0f}m, {px_str}) [{pos_lev:.0f}x Leverage]"

                if should_scratch:
                    exit_price = pos_px * (1.0 - exit_slip)
                    gross_pnl = (exit_price - entry_px) * units
                    fee = (units * exit_price) * exit_fee_rate
                    net_pnl = gross_pnl - fee
                    self.cash += max(0.0, margin_alloc + net_pnl)
                    pnl_pct = (net_pnl / max(margin_alloc, 1.0)) * 100.0
                    if net_pnl > 0:
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1

                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": "STALE_SCRATCH_LONG",
                        "price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "entry_price": round(entry_px, 4 if entry_px < 1.0 else 2),
                        "exit_price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "units": round(units, 6),
                        "gross_usd": round(units * exit_price, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "rrr": "1:1",
                        "reason": scratch_tag
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    logger.info(f"[DYNAMIC SCRATCH] LONG on {pos_sym} scratched @ {px_str} | PnL: ${net_pnl:.2f} ({scratch_tag})")
                    self.last_trade_closed_at = time.time()
                    self.positions.pop(pos_sym, None)
                    continue

            else:  # SHORT
                prev_trough = pos.get("lowest_price", entry_px)
                pos["lowest_price"] = min(prev_trough, pos_latest_candle["low"])

                # A. Check Trailing Ratchet Activation for Short (Triggered at -1.5x ATR14)
                breakeven_threshold = pos.get("breakeven_trigger", entry_px - (1.5 * pos["atr_14"]))
                if prev_trough <= breakeven_threshold:
                    if not pos.get("is_trailing", False):
                        pos["is_trailing"] = True
                        pos["stop_loss_price"] = min(pos["stop_loss_price"], entry_px * 0.998)
                        px_lbl = f"${pos['stop_loss_price']:.4f}" if pos['stop_loss_price'] < 1.0 else f"${pos['stop_loss_price']:.2f}"
                        logger.info(f"[RISK RATCHET] Breakeven locked for SHORT {pos_sym} at {px_lbl}")

                    ratchet_stop = prev_trough + (1.5 * pos["atr_14"])
                    if ratchet_stop < pos["stop_loss_price"]:
                        pos["stop_loss_price"] = ratchet_stop

                # B. Liquidation Check (Isolated Margin)
                if liq_px > 0 and pos_latest_candle["high"] >= liq_px:
                    exit_price = liq_px
                    pnl = -margin_alloc
                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": "LIQUIDATION_CLOSE_SHORT",
                        "price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "units": round(units, 6),
                        "gross_usd": round(margin_alloc, 2),
                        "fee_usd": 0.0,
                        "pnl_usd": round(pnl, 2),
                        "pnl_pct": -100.0,
                        "rrr": "0:1",
                        "reason": f"Isolated Margin Liquidation @ ${exit_price:.4f} [{pos_lev:.0f}x]" if exit_price < 1.0 else f"Isolated Margin Liquidation @ ${exit_price:.2f} [{pos_lev:.0f}x]"
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    self.consecutive_losses += 2
                    logger.warning(f"[LIQUIDATION] SHORT on {pos_sym} liquidated @ ${exit_price:.4f} | Loss: -${margin_alloc:.2f}")
                    self.last_trade_closed_at = time.time()
                    self.positions.pop(pos_sym, None)
                    continue

                # B1. Two-Stage Partial Take-Profit for Short (50% Scalp Lock at 1.2x ATR)
                tp1_threshold = pos.get("tp1_price", entry_px - (getattr(self, "tp1_atr_mult", 1.2) * pos["atr_14"]))
                stop_hit = pos_latest_candle["high"] >= pos["stop_loss_price"]
                if not stop_hit and not pos.get("tp1_hit", False) and pos_latest_candle["low"] <= tp1_threshold:
                    ratio = getattr(self, "partial_tp_ratio", 0.50)
                    close_units = pos["units"] * ratio
                    close_margin = margin_alloc * ratio
                    exit_price = tp1_threshold
                    gross_pnl = (entry_px - exit_price) * close_units
                    fee = (close_units * exit_price) * fee_rate
                    net_pnl = gross_pnl - fee

                    pos["units"] -= close_units
                    pos["margin_allocated"] -= close_margin
                    pos["tp1_hit"] = True
                    margin_alloc = pos["margin_allocated"]
                    units = pos["units"]
                    self.cash += max(0.0, close_margin + net_pnl)
                    pnl_pct = (net_pnl / max(close_margin, 1.0)) * 100.0

                    # Instantly de-risk remaining 50% runner to guaranteed breakeven (-0.2x ATR below entry)
                    pos["stop_loss_price"] = min(pos["stop_loss_price"], entry_px - (0.2 * pos["atr_14"]))
                    pos["is_trailing"] = True

                    px_str = f"${exit_price:.4f}" if exit_price < 1.0 else f"${exit_price:.2f}"
                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": "PARTIAL_TP1_SHORT",
                        "price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "entry_price": round(entry_px, 4 if entry_px < 1.0 else 2),
                        "exit_price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "units": round(close_units, 6),
                        "gross_usd": round(close_units * exit_price, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "rrr": "1:1",
                        "reason": f"TP1 Scalp Lock: {int(ratio * 100)}% closed at {px_str} (+${net_pnl:.2f}, +{pnl_pct:.1f}%). Free runner stop ratcheted to entry."
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    logger.info(f"[TP1 SCALP LOCK] SHORT {pos_sym} 50% locked @ {px_str} (+${net_pnl:.2f}) | Risk-Free Runner Active")

                # C. Take-Profit Trigger for Short
                elif not stop_hit and pos_latest_candle["low"] <= pos["take_profit_price"]:
                    exit_price = pos["take_profit_price"]
                    gross_pnl = (entry_px - exit_price) * units
                    fee = (units * exit_price) * fee_rate
                    net_pnl = gross_pnl - fee
                    self.cash += max(0.0, margin_alloc + net_pnl)
                    pnl_pct = (net_pnl / max(margin_alloc, 1.0)) * 100.0
                    self.consecutive_losses = 0

                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": "TAKE_PROFIT_CLOSE_SHORT",
                        "price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "entry_price": round(entry_px, 4 if entry_px < 1.0 else 2),
                        "exit_price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "units": round(units, 6),
                        "gross_usd": round(units * exit_price, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "rrr": f"{self.target_to_stop_ratio:.1f}:1",
                        "reason": f"Asymmetric Target Hit (+${net_pnl:.2f}, +{pnl_pct:.1f}%) [{pos_lev:.0f}x Leverage]"
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    px_str = f"${exit_price:.4f}" if exit_price < 1.0 else f"${exit_price:.2f}"
                    logger.info(f"[ASYMMETRIC WIN] TP Hit on SHORT {pos_sym} @ {px_str} | PnL: +${net_pnl:.2f}")
                    self.last_trade_closed_at = time.time()
                    self.positions.pop(pos_sym, None)
                    continue

                # D. Stop-Loss Trigger for Short
                elif pos_latest_candle["high"] >= pos["stop_loss_price"]:
                    exit_price = max(pos["stop_loss_price"], pos_px) * (1.0 + exit_slip)
                    gross_pnl = (entry_px - exit_price) * units
                    fee = (units * exit_price) * exit_fee_rate
                    net_pnl = gross_pnl - fee
                    self.cash += max(0.0, margin_alloc + net_pnl)
                    pnl_pct = (net_pnl / max(margin_alloc, 1.0)) * 100.0
                    if net_pnl > 0:
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1

                    is_trail = pos.get("is_trailing", False)
                    reason_tag = "Trailing Ratchet Stop Hit" if is_trail else f"Dynamic {self.atr_multiplier_stop:.1f}x ATR Stop Loss"
                    px_str = f"${exit_price:.4f}" if exit_price < 1.0 else f"${exit_price:.2f}"

                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": "STOP_LOSS_CLOSE_SHORT",
                        "price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "entry_price": round(entry_px, 4 if entry_px < 1.0 else 2),
                        "exit_price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "units": round(units, 6),
                        "gross_usd": round(units * exit_price, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "rrr": "2.5:1",
                        "reason": f"{reason_tag} ({px_str}) [{pos_lev:.0f}x Leverage]"
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    logger.info(f"[RISK EXIT] {reason_tag} on SHORT {pos_sym} @ {px_str} | PnL: ${net_pnl:.2f}")
                    self.last_trade_closed_at = time.time()
                    self.positions.pop(pos_sym, None)
                    continue

                # E. Dynamic Momentum Scratch (Short)
                holding_seconds = time.time() - pos.get("entry_time", time.time())
                max_hold = getattr(self, "max_holding_seconds", 3600.0)
                should_scratch = False
                scratch_tag = ""
                px_str = f"${pos_px:.4f}" if pos_px < 1.0 else f"${pos_px:.2f}"

                if not pos.get("tp1_hit", False) and holding_seconds >= max_hold:
                    should_scratch = True
                    scratch_tag = f"60m Max Holding Scratch ({holding_seconds/60:.0f}m, {px_str}) [{pos_lev:.0f}x Leverage]"

                if should_scratch:
                    exit_price = pos_px * (1.0 + exit_slip)
                    gross_pnl = (entry_px - exit_price) * units
                    fee = (units * exit_price) * exit_fee_rate
                    net_pnl = gross_pnl - fee
                    self.cash += max(0.0, margin_alloc + net_pnl)
                    pnl_pct = (net_pnl / max(margin_alloc, 1.0)) * 100.0
                    if net_pnl > 0:
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1

                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": "STALE_SCRATCH_SHORT",
                        "price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "entry_price": round(entry_px, 4 if entry_px < 1.0 else 2),
                        "exit_price": round(exit_price, 4 if exit_price < 1.0 else 2),
                        "units": round(units, 6),
                        "gross_usd": round(units * exit_price, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "rrr": "1:1",
                        "reason": scratch_tag
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    logger.info(f"[DYNAMIC SCRATCH] SHORT on {pos_sym} scratched @ {px_str} | PnL: ${net_pnl:.2f} ({scratch_tag})")
                    self.last_trade_closed_at = time.time()
                    self.positions.pop(pos_sym, None)
                    continue

    # ------------------------------------------------------------------
    # TREND mode: 4h Donchian breakout (backend/trend_strategy.py), validated in backend/backtest_trend.py
    # ------------------------------------------------------------------
    def _trend_equity(self) -> float:
        eq = self.cash
        for p in self.positions.values():
            px = float(p.get("current_price", p["entry_price"]))
            sign = 1.0 if p.get("side", "LONG").upper() == "LONG" else -1.0
            eq += p.get("margin_allocated", 0.0) + (px - p["entry_price"]) * p["units"] * sign
        return eq

    def _trend_close(self, sym: str, pos: dict, px: float, reason: str, side_tag: str):
        long = pos["side"] == "LONG"
        exit_px = px * (1.0 - self.slippage_bps / 10000.0 if long else 1.0 + self.slippage_bps / 10000.0)
        gross = (exit_px - pos["entry_price"]) * pos["units"] * (1.0 if long else -1.0)
        fee = pos["units"] * exit_px * self.taker_fee_rate
        net = gross - fee
        self.cash += max(0.0, pos["margin_allocated"] + net)
        total = net - pos.get("entry_fee", 0.0) - pos.get("funding_paid", 0.0)
        self.consecutive_losses = 0 if total > 0 else self.consecutive_losses + 1
        dec = 4 if exit_px < 1.0 else 2
        record = {
            "id": len(self.trades) + 1,
            "timestamp": int(time.time()),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": sym,
            "side": f"{side_tag}_{pos['side']}",
            "price": round(exit_px, dec),
            "entry_price": round(pos["entry_price"], dec),
            "exit_price": round(exit_px, dec),
            "units": round(pos["units"], 6),
            "gross_usd": round(pos["units"] * exit_px, 2),
            "fee_usd": round(fee, 2),
            "pnl_usd": round(net, 2),
            "pnl_pct": round(net / max(pos["margin_allocated"], 1.0) * 100.0, 2),
            "rrr": f"{total / max(pos.get('initial_risk_usd', 1e-9), 1e-9):+.2f}R",
            "reason": f"{reason} | round-trip incl. entry fee & funding: {'+' if total >= 0 else '-'}${abs(total):.2f}"
        }
        self.trades.append(record)
        self.append_trade_csv(record)
        self.positions.pop(sym, None)
        self.last_trade_closed_at = time.time()
        logger.info(f"[TREND EXIT] {pos['side']} {sym} @ {exit_px} ({reason}) net ${net:.2f}")

    def _evaluate_trend_step(self) -> dict:
        from backend.trend_strategy import compute_features, entry_signal, exit_on_close, trail_stop, position_size
        p = self.trend_params
        interval_ms = {"1d": 86_400_000, "4h": 14_400_000, "1h": 3_600_000}[p.interval]
        now = time.time()
        now_ms = int(now * 1000)
        if not hasattr(self, "trend_last_bar"):
            self.trend_last_bar = {}

        # Positions opened earlier by the 5m scalper keep their own exit rules until they close
        self._manage_scalper_positions()

        radar = []
        for sym in self.scanner_universe:
            bars = self.fetch_klines(interval=p.interval, limit=p.entry_n + p.atr_n + 40, symbol=sym)
            closed = [b for b in bars if b["timestamp"] + interval_ms <= now_ms]
            if len(closed) < p.entry_n + p.atr_n:
                continue
            live_px = float(bars[-1]["close"])
            if sym == self.symbol or not self.current_price:
                self.current_price = live_px
            f = compute_features(closed, p)
            i = len(closed) - 1
            bar_ts = closed[-1]["timestamp"]
            new_bar = self.trend_last_bar.get(sym) != bar_ts
            pos = self.positions.get(sym)

            if pos and pos.get("strategy") == "TREND":
                long = pos["side"] == "LONG"
                pos["current_price"] = live_px
                # Funding: charged at each 8h settlement crossed while the position is open
                last_f = pos.get("funding_checked_ms", now_ms)
                crossed = now_ms // 28_800_000 - last_f // 28_800_000
                if crossed > 0:
                    rate = self.fetch_derivatives_data(sym).get("funding_rate", 0.0)
                    fund = crossed * rate * pos["units"] * live_px * (1.0 if long else -1.0)
                    self.cash -= fund
                    pos["funding_paid"] = pos.get("funding_paid", 0.0) + fund
                pos["funding_checked_ms"] = now_ms
                # Resting stop-market order
                if (long and live_px <= pos["stop_loss_price"]) or (not long and live_px >= pos["stop_loss_price"]):
                    fill = min(live_px, pos["stop_loss_price"]) if long else max(live_px, pos["stop_loss_price"])
                    self._trend_close(sym, pos, fill, f"{p.stop_atr:.0f}x ATR chandelier stop", "TREND_STOP")
                    pos = None
                elif new_bar and pos.get("last_bar_ts") != bar_ts:
                    pos["last_bar_ts"] = bar_ts
                    c = float(f["close"][i])
                    pos["extreme_close"] = max(pos.get("extreme_close", c), c) if long else min(pos.get("extreme_close", c), c)
                    pos["stop_loss_price"] = trail_stop(pos["side"], pos["extreme_close"], float(f["atr"][i]), pos["stop_loss_price"], p)
                    if exit_on_close(f, i, pos["side"]):
                        self._trend_close(sym, pos, live_px, f"{p.exit_n}-bar channel exit", "TREND_EXIT")
                        pos = None

            sig = entry_signal(f, i, p)
            flag = news_guard.block_reason(sym)
            brake = flag if news_guard.enforce else None
            if flag and sig and pos is None and sym not in self.positions and new_bar:
                logger.warning(f"[NEWS BRAKE] {'skipped' if brake else 'would skip (shadow mode)'} {sig} {sym} breakout: {flag}")
            if pos is None and sym not in self.positions and new_bar and sig and not brake:
                atr = float(f["atr"][i])
                long = sig == "LONG"
                exec_px = live_px * (1.0 + self.slippage_bps / 10000.0 if long else 1.0 - self.slippage_bps / 10000.0)
                gross_used = sum(x["units"] * float(x.get("current_price", x["entry_price"])) for x in self.positions.values())
                notional = position_size(self._trend_equity(), exec_px, atr, gross_used, p, sig)
                margin = notional / p.max_gross_leverage
                fee = notional * self.taker_fee_rate
                if notional >= 10.0 and self.cash >= margin + fee:
                    units = notional / exec_px
                    stop = exec_px - p.stop_atr * atr if long else exec_px + p.stop_atr * atr
                    self.cash -= margin + fee
                    self.positions[sym] = {
                        "symbol": sym, "side": sig, "strategy": "TREND", "units": units,
                        "leverage": p.max_gross_leverage, "margin_allocated": margin,
                        "entry_price": exec_px, "current_price": exec_px, "entry_time": time.time(),
                        "highest_price": exec_px, "lowest_price": exec_px, "extreme_close": exec_px,
                        "stop_loss_price": stop, "take_profit_price": 0.0, "tp1_hit": True,
                        "atr_14": atr, "is_trailing": True, "liquidation_price": 0.0,
                        "entry_fee": fee, "funding_paid": 0.0, "funding_checked_ms": now_ms,
                        "initial_risk_usd": units * abs(exec_px - stop), "last_bar_ts": bar_ts,
                        "execution_type": "TAKER_MARKET",
                    }
                    dec = 4 if exec_px < 1.0 else 2
                    record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": sym,
                        "side": "BUY_LONG" if long else "SELL_SHORT",
                        "price": round(exec_px, dec),
                        "entry_price": round(exec_px, dec),
                        "exit_price": None,
                        "units": round(units, 6),
                        "gross_usd": round(margin, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": 0.0,
                        "pnl_pct": 0.0,
                        "rrr": "trend",
                        "reason": (f"TREND {p.interval} {p.entry_n}-bar breakout {sig} | stop ${stop:.{dec}f} "
                                   f"({p.stop_atr:.0f}x ATR) | risk {p.risk_pct*100:.1f}% equity, notional ${notional:.0f}")
                    }
                    self.trades.append(record)
                    self.append_trade_csv(record)
                    logger.info(f"[TREND ENTRY] {sig} {sym} @ {exec_px} stop {stop} notional ${notional:.0f}")

            self.trend_last_bar[sym] = bar_ts

            upper, lower = float(f["upper"][i]), float(f["lower"][i])
            dist_up = (upper - live_px) / live_px * 100.0
            active = sym in self.positions
            radar.append({
                "symbol": sym,
                "price": round(live_px, 4 if live_px < 1.0 else 2),
                "change_5m": round((live_px / float(bars[-1]["open"]) - 1) * 100.0, 2),
                "htf_regime": "TREND_BULL" if live_px > upper else ("TREND_BEAR" if live_px < lower else "RANGE_BOUND"),
                "is_squeeze": False,
                "cvd_mom": 0.0,
                "depth_ratio": 1.0,
                "score": round(max(0.0, min(100.0, 100.0 - max(0.0, dist_up) * 10.0)), 1),
                "choice": "BUY_LONG" if (sig == "LONG" and not active) else ("SELL_SHORT" if (sig == "SHORT" and not active) else "WAIT"),
                "news_score": 0.0,
                "top_catalyst": (f"NEWS {'BRAKE' if brake else 'FLAG'}: {flag}" if flag else
                                 f"{p.entry_n}-bar high ${upper:.{4 if upper < 1 else 2}f} ({dist_up:+.1f}% away)"),
                "is_active_trade": active,
                "status_label": "ACTIVE TREND POSITION" if active else ("NEWS BRAKE" if brake else "SCANNING"),
                "consensus_status": "TREND_MODE",
            })

        radar.sort(key=lambda r: (r["is_active_trade"], r["score"]), reverse=True)
        self.radar_scan_results = radar
        self.last_checked_at = now

        total_equity = self._trend_equity()
        first_pos = self.position
        snap_px = float(first_pos.get("current_price", first_pos["entry_price"])) if first_pos else self.current_price
        self.hourly_snapshots.append({
            "timestamp": int(now),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": first_pos.get("symbol", self.symbol) if first_pos else self.symbol,
            "price": round(snap_px, 4 if snap_px < 1.0 else 2),
            "equity": round(total_equity, 2),
            "cash": round(self.cash, 2),
            "htf_regime": "TREND_MODE",
            "atr_14": 0.0,
            "funding_rate": self.current_funding_rate,
            "has_position": len(self.positions) > 0,
            "active_positions_count": len(self.positions)
        })
        self.save_state()
        return {"mode": "TREND", "total_equity": total_equity, "positions": list(self.positions.values()),
                "position": self.position, "action": "hold", "confidence": 1.0, "gated": False}

    def set_strategy_mode(self, mode: str) -> dict:
        mode = mode.upper()
        if mode not in ("TREND", "SCALPER"):
            return {"status": "error", "message": "mode must be TREND or SCALPER"}
        with self.lock:
            self.strategy_mode = mode
            self.save_state()
        return {"status": "ok", "strategy_mode": mode}

    def evaluate_step(self) -> dict:
        # Serialized: the background loop and API-triggered steps must never close the same position twice
        with self.lock:
            return self._evaluate_step_unlocked()

    def _evaluate_step_unlocked(self) -> dict:
        """
        Institutional Decision & Execution Cycle:
        1. Higher Timeframe (4h) Macro Regime classification
        2. Derivatives Perpetual Funding Skew inspection
        3. ATR Volatility Dynamic Stops & Asymmetric 2.5:1 RRR
        4. Trailing Ratchet management
        """
        if getattr(self, "strategy_mode", "TREND") == "TREND":
            return self._evaluate_trend_step()
        active_sym = self.position.get("symbol", self.symbol) if self.position else self.symbol
        ltf_candles = self.fetch_klines(interval=self.interval, limit=50, symbol=active_sym)
        if not ltf_candles:
            return {"error": "no_market_data"}

        latest_candle = ltf_candles[-1]
        self.current_price = latest_candle["close"]
        self.last_checked_at = time.time()
        self.last_candle_timestamp = latest_candle["timestamp"]

        # Compute LTF Indicators & ATR
        atr_14, atr_exp, sma_fast, sma_slow, rsi, vol_spike, cvd_mom = self._compute_atr_and_indicators(ltf_candles)
        self.current_atr = round(atr_14, 2)

        # 1. Multi-Timeframe Macro Regime
        self.current_htf_regime = self.classify_htf_macro_regime(symbol=active_sym)

        # 2. Derivatives Data
        deriv = self.fetch_derivatives_data(symbol=active_sym)
        self.current_funding_rate = deriv["funding_rate"]
        self.current_funding_skew = deriv["funding_skew"]

        fee_rate = self.maker_fee_rate if self.use_maker_execution else self.taker_fee_rate
        slippage = 0.0 if self.use_maker_execution else (self.slippage_bps / 10000.0)
        exit_fee_rate = self.taker_fee_rate
        exit_slip = self.slippage_bps / 10000.0

        # ==========================================
        # 3. ACTIVE POSITIONS MANAGEMENT (ATR Dynamic & Liquidation Guard)
        # ==========================================
        self._manage_scalper_positions()

        # ==========================================
        # 4. DECISION ENGINE EVALUATION
        # ==========================================
        # Hard Circuit Breaker: 100% Cash in TOXIC CHOP
        if self.current_htf_regime == "TOXIC_CHOP":
            decision = {
                "choice": "hold",
                "confidence": 0.98,
                "reasoning": "HTF Circuit Breaker: 4h TOXIC_CHOP liquidation regime. Enforcing 100% CASH.",
                "engine": "macro-regime-breaker"
            }
            choice = "hold"
            confidence = 0.98
            reasoning = decision["reasoning"]
            engine = "macro-regime-breaker"
        else:
            market_state = {
                "ticker": active_sym,
                "price": round(self.current_price, 4 if self.current_price < 1.0 else 2),
                "rsi": round(rsi, 1),
                "sma_fast": round(sma_fast, 2),
                "sma_slow": round(sma_slow, 2),
                "volume_spike": round(vol_spike, 2),
                "atr_14": round(atr_14, 2),
                "atr_expansion": round(atr_exp, 2),
                "funding_rate": self.current_funding_rate,
                "cvd_delta": round(cvd_mom, 2),
                "htf_regime": self.current_htf_regime
            }

            try:
                from backend.news_engine import news_engine
                from backend.neural_dream import intelligent_dream_trainer
                news_state = news_engine.get_ticker_sentiment(active_sym)
                self.current_news_sentiment = news_state.get("score", 0.0)
                self.current_news_label = news_state.get("label", "Neutral")
                self.top_headline = news_state.get("top_catalyst", "")

                neural_res = intelligent_dream_trainer.infer(market_state, news_state)
                choice = neural_res.get("choice", "hold")
                confidence = float(neural_res.get("confidence", 0.70))
                reasoning = neural_res.get("reasoning", "")
                engine = "neural-market-intelligence"
                self.latest_academic_synthesis = neural_res.get("academic_synthesis", "")
                self.latest_research_metrics = neural_res.get("research_metrics", {})

                # Hard Lock Stop Loss at 1.35x ATR to preserve positive expectancy
                self.atr_multiplier_stop = 1.35
                if "dynamic_tp_atr" in neural_res:
                    self.target_to_stop_ratio = max(2.0, neural_res["dynamic_tp_atr"] / max(0.1, self.atr_multiplier_stop))
            except Exception as e:
                logger.warning(f"Neural inference fallback: {e}")
                if self.decision_callback:
                    decision = self.decision_callback(market_state)
                else:
                    decision = {"choice": "hold", "confidence": 0.5, "reasoning": "Hold fallback", "engine": "fallback"}
                choice = decision.get("choice", "hold")
                confidence = float(decision.get("confidence", 0.0))
                reasoning = decision.get("reasoning", "")
                engine = decision.get("engine", "laya")

        is_gated = (confidence < self.confidence_gate)

        # Audit Log
        self.audit_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "price": self.current_price,
            "rsi": round(rsi, 1),
            "htf_regime": self.current_htf_regime,
            "atr_14": round(atr_14, 2),
            "funding": f"{self.current_funding_rate*100:+.3f}%",
            "choice": choice,
            "confidence": round(confidence, 4),
            "gated": is_gated,
            "reasoning": reasoning
        })
        if len(self.audit_log) > 200:
            self.audit_log = self.audit_log[-200:]

        # ==========================================
        # 5. ORDER EXECUTION WITH MULTI-POSITION SLOTS & COMPOUNDING
        # ==========================================
        # Anti-Churn Armor: Enforce 2-min cooldown
        cooldown_elapsed = time.time() - getattr(self, "last_trade_closed_at", 0.0)
        in_cooldown = cooldown_elapsed < getattr(self, "trade_cooldown_seconds", 600.0)
        available_slots = getattr(self, "max_concurrent_positions", 3) - len(self.positions)

        if not in_cooldown and available_slots > 0:
            # -------------------------------------------------------------
            # 1. AUTONOMOUS MULTI-TICKER MARKET SCANNER (Auto-Pick Mode)
            # -------------------------------------------------------------
            if getattr(self, "scanner_mode", True):
                radar = self.scan_market_universe()
                open_syms = set(self.positions.keys())
                # High-Conviction Institutional Filter: Dual Consensus + Score >= 88.0 + Range-Bound Squeeze Guard
                dual_candidates = [
                    r for r in radar 
                    if r.get("dual_consensus_achieved") 
                    and r.get("choice") in ["BUY_LONG", "SELL_SHORT"] 
                    and r.get("score", 0) >= 88.0 
                    and (r.get("htf_regime") != "RANGE_BOUND" or r.get("is_squeeze") or r.get("score", 0) >= 92.0)
                    and r.get("symbol") not in open_syms
                ]
                candidates = dual_candidates
                if candidates:
                    best = candidates[0]
                    target_sym = best["symbol"]
                    target_choice = "buy" if best["choice"] == "BUY_LONG" else "sell"
                    # Use unrounded values: display rounding (2dp) shifts entries on ~$1 coins by up to 0.5%
                    target_px = float(best.get("price_raw", best["price"]))
                    target_atr = float(best.get("atr_raw", best["atr_14"]))

                    # Dynamic Leverage modulated by Model A Macro Governor
                    allowed_lev = float(best.get("recommended_leverage", self.leverage))
                    active_leverage = min(float(getattr(self, "leverage", 5.0)), allowed_lev, 5.0)

                    capital_per_slot = min(self.cash / available_slots, self.cash * 0.80)
                    stop_dist = max(self.atr_multiplier_stop * target_atr, target_px * 0.008)
                    stop_dist_pct = stop_dist / max(target_px, 1e-4)
                    margin_alloc, notional, risk_pct = self.compute_compounding_allocation(capital_per_slot, stop_dist_pct)

                    entry_notional = margin_alloc * active_leverage
                    fee = entry_notional * fee_rate
                    if margin_alloc >= 15.0 and self.cash >= margin_alloc + fee:
                        exec_price = target_px * (1.0 + slippage if target_choice == "buy" else 1.0 - slippage)
                        units = entry_notional / exec_price

                        # Track Maker Fee Savings
                        fee_saved = 0.0
                        if self.use_maker_execution:
                            taker_fee_standard = entry_notional * self.taker_fee_rate
                            fee_saved = max(0.0, taker_fee_standard - fee)
                            self.total_maker_fee_savings_usd = getattr(self, "total_maker_fee_savings_usd", 0.0) + fee_saved

                        tp1_dist = getattr(self, "tp1_atr_mult", 1.2) * target_atr

                        if target_choice == "buy":
                            stop_loss_price = exec_price - stop_dist
                            take_profit_price = exec_price + (self.target_to_stop_ratio * stop_dist)
                            tp1_price = exec_price + tp1_dist
                            breakeven_trigger = exec_price + (1.5 * target_atr)
                            liq_price = exec_price * (1.0 - (1.0 / max(1.0, active_leverage)) + self.maintenance_margin_rate) if active_leverage > 1.0 else 0.0
                            side_label = "LONG"
                            trade_side = "BUY_LONG"
                        else:
                            stop_loss_price = exec_price + stop_dist
                            take_profit_price = exec_price - (self.target_to_stop_ratio * stop_dist)
                            tp1_price = exec_price - tp1_dist
                            breakeven_trigger = exec_price - (1.5 * target_atr)
                            liq_price = exec_price * (1.0 + (1.0 / max(1.0, active_leverage)) - self.maintenance_margin_rate) if active_leverage > 1.0 else 0.0
                            side_label = "SHORT"
                            trade_side = "SELL_SHORT"

                        self.cash -= margin_alloc + fee
                        new_pos = {
                            "symbol": target_sym,
                            "side": side_label,
                            "units": units,
                            "leverage": active_leverage,
                            "margin_allocated": margin_alloc,
                            "entry_price": exec_price,
                            "current_price": exec_price,
                            "entry_time": time.time(),
                            "highest_price": exec_price,
                            "lowest_price": exec_price,
                            "stop_loss_price": stop_loss_price,
                            "take_profit_price": take_profit_price,
                            "tp1_price": tp1_price,
                            "tp1_hit": False,
                            "initial_units": units,
                            "initial_margin": margin_alloc,
                            "depth_ratio": best.get("depth_ratio", 1.0),
                            "vpin_toxicity": best.get("vpin_toxicity", "CLEAN"),
                            "breakeven_trigger": breakeven_trigger,
                            "liquidation_price": liq_price,
                            "atr_14": target_atr,
                            "is_trailing": False,
                            "execution_type": "MAKER_POST_ONLY" if self.use_maker_execution else "TAKER_MARKET"
                        }
                        self.positions[target_sym] = new_pos

                        savings_tag = f" | Saved ${fee_saved:.2f} fees" if fee_saved > 0 else ""
                        trade_record = {
                            "id": len(self.trades) + 1,
                            "timestamp": int(time.time()),
                            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": target_sym,
                            "side": trade_side,
                            "price": round(exec_price, 4 if exec_price < 1.0 else 2),
                            "entry_price": round(exec_price, 4 if exec_price < 1.0 else 2),
                            "exit_price": None,
                            "units": round(units, 6),
                            "gross_usd": round(margin_alloc, 2),
                            "fee_usd": round(fee, 2),
                            "fee_saved_usd": round(fee_saved, 3),
                            "pnl_usd": 0.0,
                            "pnl_pct": 0.0,
                            "rrr": f"{self.target_to_stop_ratio:.1f}:1",
                            "reason": f"DUAL-MODEL {target_sym} (Score {best['score']}/100, OFI Depth: {best.get('depth_ratio', 1.0)}x, VPIN: {best.get('vpin_toxicity', 'CLEAN')}){savings_tag} [{active_leverage:.0f}x Iso]"
                        }
                        self.trades.append(trade_record)
                        self.append_trade_csv(trade_record)
                        logger.info(f"[MULTI-PORTFOLIO ENTRY] Opened {side_label} on {target_sym} @ ${exec_price:.4f} (TP1: ${tp1_price:.4f}, OFI: {best.get('depth_ratio', 1.0)}x, Active: {len(self.positions)}/{getattr(self, 'max_concurrent_positions', 3)})")

            # -------------------------------------------------------------
            # 2. SINGLE-TICKER FALLBACK ENTRY (when scanner_mode is False)
            # -------------------------------------------------------------
            elif not is_gated and self.current_htf_regime != "TOXIC_CHOP" and self.symbol not in self.positions:
                stop_dist = max(self.atr_multiplier_stop * atr_14, self.current_price * 0.008)
                stop_dist_pct = stop_dist / max(self.current_price, 1e-4)
                capital_per_slot = min(self.cash / available_slots, self.cash * 0.80)
                margin_alloc, notional, risk_pct = self.compute_compounding_allocation(capital_per_slot, stop_dist_pct)

                if choice == "buy" and margin_alloc >= 15.0 and self.cash >= margin_alloc * (1.0 + self.leverage * fee_rate):
                    exec_price = self.current_price * (1.0 + slippage)
                    fee = margin_alloc * self.leverage * fee_rate
                    units = (margin_alloc * self.leverage) / exec_price

                    stop_loss_price = exec_price - stop_dist
                    take_profit_price = exec_price + (self.target_to_stop_ratio * stop_dist)
                    breakeven_trigger = exec_price + (1.5 * atr_14)
                    liq_price = exec_price * (1.0 - (1.0 / max(1.0, self.leverage)) + self.maintenance_margin_rate) if self.leverage > 1.0 else 0.0

                    self.cash -= margin_alloc + fee
                    self.positions[self.symbol] = {
                        "symbol": self.symbol,
                        "side": "LONG",
                        "units": units,
                        "leverage": self.leverage,
                        "margin_allocated": margin_alloc,
                        "entry_price": exec_price,
                        "current_price": exec_price,
                        "entry_time": time.time(),
                        "highest_price": exec_price,
                        "lowest_price": exec_price,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                        "breakeven_trigger": breakeven_trigger,
                        "liquidation_price": liq_price,
                        "atr_14": atr_14,
                        "is_trailing": False,
                        "execution_type": "MAKER_POST_ONLY" if self.use_maker_execution else "TAKER_MARKET"
                    }

                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": self.symbol,
                        "side": "BUY_LONG",
                        "price": round(exec_price, 2),
                        "entry_price": round(exec_price, 2),
                        "exit_price": None,
                        "units": round(units, 6),
                        "gross_usd": round(margin_alloc, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": 0.0,
                        "pnl_pct": 0.0,
                        "rrr": f"{self.target_to_stop_ratio:.1f}:1",
                        "reason": f"LONG {self.leverage:.0f}x | SL: ${stop_loss_price:.2f} | TP: ${take_profit_price:.2f} | Liq: ${liq_price:.2f}"
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    logger.info(f"[ENTRY LONG] Bought {units:.4f} {self.symbol} @ ${exec_price:.2f} [{self.leverage:.0f}x isolated, margin: ${margin_alloc:.2f}]")

                elif choice == "sell" and margin_alloc >= 15.0 and self.cash >= margin_alloc * (1.0 + self.leverage * fee_rate):
                    exec_price = self.current_price * (1.0 - slippage)
                    fee = margin_alloc * self.leverage * fee_rate
                    units = (margin_alloc * self.leverage) / exec_price

                    stop_loss_price = exec_price + stop_dist
                    take_profit_price = exec_price - (self.target_to_stop_ratio * stop_dist)
                    breakeven_trigger = exec_price - (1.5 * atr_14)
                    liq_price = exec_price * (1.0 + (1.0 / max(1.0, self.leverage)) - self.maintenance_margin_rate) if self.leverage > 1.0 else 0.0

                    self.cash -= margin_alloc + fee
                    self.positions[self.symbol] = {
                        "symbol": self.symbol,
                        "side": "SHORT",
                        "units": units,
                        "leverage": self.leverage,
                        "margin_allocated": margin_alloc,
                        "entry_price": exec_price,
                        "current_price": exec_price,
                        "entry_time": time.time(),
                        "highest_price": exec_price,
                        "lowest_price": exec_price,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                        "breakeven_trigger": breakeven_trigger,
                        "liquidation_price": liq_price,
                        "atr_14": atr_14,
                        "is_trailing": False,
                        "execution_type": "MAKER_POST_ONLY" if self.use_maker_execution else "TAKER_MARKET"
                    }

                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": self.symbol,
                        "side": "SELL_SHORT",
                        "price": round(exec_price, 2),
                        "entry_price": round(exec_price, 2),
                        "exit_price": None,
                        "units": round(units, 6),
                        "gross_usd": round(margin_alloc, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": 0.0,
                        "pnl_pct": 0.0,
                        "rrr": f"{self.target_to_stop_ratio:.1f}:1",
                        "reason": f"SHORT {self.leverage:.0f}x | SL: ${stop_loss_price:.2f} | TP: ${take_profit_price:.2f} | Liq: ${liq_price:.2f}"
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    logger.info(f"[ENTRY SHORT] Sold Short {units:.4f} {self.symbol} @ ${exec_price:.2f} [{self.leverage:.0f}x isolated, margin: ${margin_alloc:.2f}]")

        # B. Model Discretionary Exits across active positions
        for pos_sym, pos in list(self.positions.items()):
            side = pos.get("side", "LONG").upper()
            margin_alloc = pos.get("margin_allocated", (pos["units"] * pos["entry_price"]) / max(1.0, pos.get("leverage", self.leverage)))
            holding_seconds = time.time() - pos.get("entry_time", time.time())
            can_model_exit = holding_seconds >= getattr(self, "min_holding_seconds", 300.0)
            is_emergency_exit = (self.current_htf_regime == "TOXIC_CHOP") or (getattr(self, "dual_model_telemetry", {}).get("veto_active", False))
            opposite_signal = (side == "LONG" and choice == "sell" and confidence >= 0.82) or (side == "SHORT" and choice == "buy" and confidence >= 0.82)

            if is_emergency_exit or (can_model_exit and opposite_signal and pos_sym == self.symbol):
                pos_cur_px = float(pos.get("current_price", pos.get("entry_price", 0.0)))
                exec_price = pos_cur_px * (1.0 - exit_slip if side == "LONG" else 1.0 + exit_slip)
                gross_pnl = (exec_price - pos["entry_price"]) * pos["units"] if side == "LONG" else (pos["entry_price"] - exec_price) * pos["units"]
                fee = (pos["units"] * exec_price) * exit_fee_rate
                net_pnl = gross_pnl - fee
                self.cash += max(0.0, margin_alloc + net_pnl)
                pnl_pct = (net_pnl / max(margin_alloc, 1.0)) * 100.0
                if net_pnl > 0:
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1

                exit_reason = f"Macro Emergency Exit: {reasoning}" if is_emergency_exit else f"High-Conviction Reversal ({confidence*100:.0f}%): {reasoning}"
                px_str = f"${exec_price:.4f}" if exec_price < 1.0 else f"${exec_price:.2f}"
                trade_record = {
                    "id": len(self.trades) + 1,
                    "timestamp": int(time.time()),
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": pos_sym,
                    "side": "MODEL_CLOSE_LONG" if side == "LONG" else "MODEL_CLOSE_SHORT",
                    "price": round(exec_price, 4 if exec_price < 1.0 else 2),
                    "units": round(pos["units"], 6),
                    "gross_usd": round(pos["units"] * exec_price, 2),
                    "fee_usd": round(fee, 2),
                    "pnl_usd": round(net_pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "rrr": f"{self.target_to_stop_ratio:.1f}:1",
                    "reason": exit_reason
                }
                self.trades.append(trade_record)
                self.append_trade_csv(trade_record)
                logger.info(f"[MODEL EXIT] Closed {side} {pos_sym} @ {px_str} | PnL: ${net_pnl:.2f} ({exit_reason})")
                self.last_trade_closed_at = time.time()
                self.positions.pop(pos_sym, None)

        # Total Portfolio Equity Snapshot across all positions
        total_pos_val = 0.0
        for p_sym, p in self.positions.items():
            p_side = p.get("side", "LONG").upper()
            entry = p["entry_price"]
            units = p["units"]
            pos_cur_px = float(p.get("current_price", entry))
            margin = p.get("margin_allocated", (units * entry) / max(1.0, p.get("leverage", self.leverage)))
            fee_rate = self.maker_fee_rate if self.use_maker_execution else self.taker_fee_rate
            if p_side == "LONG":
                gross_pnl = (pos_cur_px - entry) * units
            else:
                gross_pnl = (entry - pos_cur_px) * units
            est_fee = (units * pos_cur_px) * fee_rate
            total_pos_val += max(0.0, margin + gross_pnl - est_fee)

        total_equity = self.cash + total_pos_val
        first_pos = self.position
        snapshot_px = float(first_pos.get("current_price", first_pos["entry_price"])) if first_pos else self.current_price

        self.hourly_snapshots.append({
            "timestamp": int(time.time()),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": first_pos.get("symbol", self.symbol) if first_pos else self.symbol,
            "price": round(snapshot_px, 4 if snapshot_px < 1.0 else 2),
            "equity": round(total_equity, 2),
            "cash": round(self.cash, 2),
            "htf_regime": self.current_htf_regime,
            "atr_14": round(atr_14, 2),
            "funding_rate": self.current_funding_rate,
            "has_position": len(self.positions) > 0,
            "active_positions_count": len(self.positions)
        })
        self.save_state()

        return {
            "price": self.current_price,
            "htf_regime": self.current_htf_regime,
            "atr_14": round(atr_14, 2),
            "funding_rate": self.current_funding_rate,
            "action": choice,
            "confidence": confidence,
            "gated": is_gated,
            "total_equity": total_equity,
            "positions": list(self.positions.values()),
            "position": self.position
        }

    def _run_loop(self):
        logger.info(f"Institutional 24h loop active. Target: {self.target_duration_hours} hours.")
        while self.is_running and not self.stop_event.is_set():
            try:
                elapsed_hours = (time.time() - self.started_at) / 3600.0 if self.started_at > 0 else 0
                if elapsed_hours >= self.target_duration_hours:
                    logger.info(f"Target duration of {self.target_duration_hours}h reached.")
                    self.stop()
                    break

                self._apply_recurring_deposit()
                self.evaluate_step()
            except Exception as e:
                logger.error(f"Error in institutional loop: {e}", exc_info=True)

            self.stop_event.wait(timeout=self.poll_seconds)

    def deploy_neural_model(self) -> dict:
        with self.lock:
            self.active_policy_source = "NEURAL_MARKET_INTELLIGENCE"
            self.active_formula = "Deep PyTorch 18-D Neural Policy (Behavioral Economics • Lopez de Prado VPIN • Macro Regimes • News Flow)"
            self.save_state()
        from backend.neural_dream import intelligent_dream_trainer
        return {
            "status": "deployed",
            "active_model_type": self.active_policy_source,
            "active_policy_source": self.active_policy_source,
            "active_formula": self.active_formula,
            "sl_atr": self.atr_multiplier_stop,
            "tp_atr": round(self.atr_multiplier_stop * self.target_to_stop_ratio, 2),
            "factor_weights": intelligent_dream_trainer.learned_factor_weights,
            "metadata": intelligent_dream_trainer.model_metadata,
            "message": "Successfully deployed Deep Neural Market Intelligence Model with PhD Behavioral Research to live trading desk!"
        }

    def open_manual_position(self, side: str, margin_amount: float = None) -> dict:
        with self.lock:
            if self.position is not None:
                return {"status": "error", "message": f"Position already open on {self.position.get('side')} {self.symbol}."}
            side = side.upper()
            if side not in ["LONG", "SHORT"]:
                return {"status": "error", "message": "Position side must be LONG or SHORT."}

            ltf_candles = self.fetch_klines(interval=self.interval, limit=30)
            if ltf_candles:
                atr_14, _, _, _, _, _, _ = self._compute_atr_and_indicators(ltf_candles)
                self.current_price = ltf_candles[-1]["close"]
            else:
                atr_14 = self.current_atr if self.current_atr > 0 else (self.current_price * 0.015)

            stop_dist = max(self.atr_multiplier_stop * atr_14, self.current_price * 0.008)
            stop_dist_pct = stop_dist / max(self.current_price, 1e-4)

            if margin_amount and float(margin_amount) > 0:
                margin_alloc = min(float(margin_amount), self.cash * 0.95)
            elif self.auto_compounding:
                margin_alloc, _, _ = self.compute_compounding_allocation(self.cash, stop_dist_pct)
            else:
                margin_alloc = min(self.cash * self.allocation_pct, self.cash * 0.95)

            if margin_alloc < 10.0:
                return {"status": "error", "message": f"Insufficient cash for trade margin (allocated: ${margin_alloc:.2f})."}

            fee_rate = self.maker_fee_rate if self.use_maker_execution else self.taker_fee_rate
            slippage = 0.0 if self.use_maker_execution else (self.slippage_bps / 10000.0)

            notional = margin_alloc * self.leverage
            fee = notional * fee_rate

            if side == "LONG":
                exec_price = self.current_price * (1.0 + slippage)
                units = notional / exec_price
                stop_loss_price = exec_price - stop_dist
                take_profit_price = exec_price + (self.target_to_stop_ratio * stop_dist)
                breakeven_trigger = exec_price + (1.5 * atr_14)
                liq_price = exec_price * (1.0 - (1.0 / max(1.0, self.leverage)) + self.maintenance_margin_rate) if self.leverage > 1.0 else 0.0
            else:  # SHORT
                exec_price = self.current_price * (1.0 - slippage)
                units = notional / exec_price
                stop_loss_price = exec_price + stop_dist
                take_profit_price = exec_price - (self.target_to_stop_ratio * stop_dist)
                breakeven_trigger = exec_price - (1.5 * atr_14)
                liq_price = exec_price * (1.0 + (1.0 / max(1.0, self.leverage)) - self.maintenance_margin_rate) if self.leverage > 1.0 else 0.0

            self.cash -= margin_alloc + fee
            self.position = {
                "symbol": self.symbol,
                "side": side,
                "units": units,
                "leverage": self.leverage,
                "margin_allocated": margin_alloc,
                "entry_price": exec_price,
                "entry_time": time.time(),
                "highest_price": exec_price,
                "lowest_price": exec_price,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "breakeven_trigger": breakeven_trigger,
                "liquidation_price": liq_price,
                "atr_14": atr_14,
                "is_trailing": False,
                "execution_type": "MANUAL_LIMIT" if self.use_maker_execution else "MANUAL_MARKET"
            }

            trade_record = {
                "id": len(self.trades) + 1,
                "timestamp": int(time.time()),
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": self.symbol,
                "side": f"MANUAL_{side}",
                "price": round(exec_price, 2),
                "units": round(units, 6),
                "gross_usd": round(margin_alloc, 2),
                "fee_usd": round(fee, 2),
                "pnl_usd": 0.0,
                "pnl_pct": 0.0,
                "rrr": f"{self.target_to_stop_ratio:.1f}:1",
                "reason": f"Manual {side} [{self.leverage:.0f}x Isolated] | SL: ${stop_loss_price:.2f} | TP: ${take_profit_price:.2f}"
            }
            self.trades.append(trade_record)
            self.append_trade_csv(trade_record)
            self.save_state()
            logger.info(f"[MANUAL ORDER] Opened {side} on {self.symbol} @ ${exec_price:.2f} [{self.leverage:.0f}x]")
            return {"status": "opened", "position": self.position, "trade": trade_record}

    def close_manual_position(self, symbol: Optional[str] = None) -> dict:
        with self.lock:
            if not self.positions:
                return {"status": "error", "message": "No active positions to close."}

            fee_rate = self.taker_fee_rate
            slippage = self.slippage_bps / 10000.0

            # Close ALL active positions
            if symbol and symbol.upper() == "ALL":
                closed_trades = []
                for pos_sym, pos in list(self.positions.items()):
                    side = pos.get("side", "LONG").upper()
                    units = pos["units"]
                    entry_px = pos["entry_price"]
                    margin_alloc = pos.get("margin_allocated", (units * entry_px) / max(1.0, pos.get("leverage", self.leverage)))

                    pos_candles = self.fetch_klines(interval=self.interval, limit=5, symbol=pos_sym)
                    pos_cur_px = float(pos_candles[-1]["close"]) if pos_candles else float(pos.get("current_price", entry_px))
                    exec_price = pos_cur_px * (1.0 - slippage if side == "LONG" else 1.0 + slippage)
                    gross_pnl = (exec_price - entry_px) * units if side == "LONG" else (entry_px - exec_price) * units
                    fee = (units * exec_price) * fee_rate
                    net_pnl = gross_pnl - fee
                    self.cash += max(0.0, margin_alloc + net_pnl)
                    pnl_pct = (net_pnl / max(margin_alloc, 1.0)) * 100.0

                    px_str = f"${exec_price:.4f}" if exec_price < 1.0 else f"${exec_price:.2f}"
                    trade_record = {
                        "id": len(self.trades) + 1,
                        "timestamp": int(time.time()),
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": pos_sym,
                        "side": f"MANUAL_CLOSE_{side}",
                        "price": round(exec_price, 4 if exec_price < 1.0 else 2),
                        "units": round(units, 6),
                        "gross_usd": round(units * exec_price, 2),
                        "fee_usd": round(fee, 2),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "rrr": f"{self.target_to_stop_ratio:.1f}:1",
                        "reason": f"Manual Portfolio Flush: {side} {pos_sym} @ {px_str} ({'+' if net_pnl >= 0 else ''}${net_pnl:.2f})"
                    }
                    self.trades.append(trade_record)
                    self.append_trade_csv(trade_record)
                    self.positions.pop(pos_sym, None)
                    closed_trades.append(trade_record)

                self.last_trade_closed_at = time.time()
                self.save_state()
                logger.info(f"[MANUAL CLOSE ALL] Closed {len(closed_trades)} positions. Remaining cash: ${self.cash:.2f}")
                return {"status": "closed_all", "trades": closed_trades, "remaining_cash": self.cash}

            # Close specific target position
            target_sym = None
            if symbol and symbol.upper() in self.positions:
                target_sym = symbol.upper()
            elif self.symbol in self.positions:
                target_sym = self.symbol
            elif self.positions:
                target_sym = next(iter(self.positions.keys()))

            if not target_sym or target_sym not in self.positions:
                return {"status": "error", "message": f"No active position for '{symbol or 'current symbol'}'."}

            pos = self.positions[target_sym]
            pos_sym = target_sym
            side = pos.get("side", "LONG").upper()
            units = pos["units"]
            entry_px = pos["entry_price"]
            margin_alloc = pos.get("margin_allocated", (units * entry_px) / max(1.0, pos.get("leverage", self.leverage)))

            pos_candles = self.fetch_klines(interval=self.interval, limit=5, symbol=pos_sym)
            if pos_candles:
                pos_cur_px = float(pos_candles[-1]["close"])
            else:
                pos_cur_px = float(pos.get("current_price", entry_px))

            exec_price = pos_cur_px * (1.0 - slippage if side == "LONG" else 1.0 + slippage)
            if side == "LONG":
                gross_pnl = (exec_price - entry_px) * units
            else:
                gross_pnl = (entry_px - exec_price) * units

            fee = (units * exec_price) * fee_rate
            net_pnl = gross_pnl - fee
            self.cash += max(0.0, margin_alloc + net_pnl)
            pnl_pct = (net_pnl / max(margin_alloc, 1.0)) * 100.0

            if net_pnl > 0:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

            px_str = f"${exec_price:.4f}" if exec_price < 1.0 else f"${exec_price:.2f}"
            trade_record = {
                "id": len(self.trades) + 1,
                "timestamp": int(time.time()),
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": pos_sym,
                "side": f"MANUAL_CLOSE_{side}",
                "price": round(exec_price, 4 if exec_price < 1.0 else 2),
                "units": round(units, 6),
                "gross_usd": round(units * exec_price, 2),
                "fee_usd": round(fee, 2),
                "pnl_usd": round(net_pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "rrr": f"{self.target_to_stop_ratio:.1f}:1",
                "reason": f"Manual Close of {side} {pos_sym} @ {px_str} ({'+' if net_pnl >= 0 else ''}${net_pnl:.2f})"
            }
            self.trades.append(trade_record)
            self.append_trade_csv(trade_record)
            self.last_trade_closed_at = time.time()
            self.positions.pop(pos_sym, None)
            self.save_state()
            logger.info(f"[MANUAL CLOSE] Closed {side} on {pos_sym} @ {px_str} | PnL: ${net_pnl:.2f}")
            return {"status": "closed", "trade": trade_record, "symbol": pos_sym, "remaining_cash": self.cash}

    def set_max_concurrent_positions(self, max_pos: int) -> dict:
        with self.lock:
            self.max_concurrent_positions = max(1, min(5, int(max_pos)))
            self.save_state()
            logger.info(f"Updated Paper Trader max concurrent positions to {self.max_concurrent_positions}")
            return {"status": "success", "max_concurrent_positions": self.max_concurrent_positions}

    def set_leverage(self, leverage: float) -> dict:
        with self.lock:
            self.leverage = float(np.clip(float(leverage), 1.0, 5.0))
            self.save_state()
            logger.info(f"Updated Paper Trader leverage to {self.leverage:.0f}x")
            return {"status": "success", "leverage": self.leverage}

    def set_compounding(self, enabled: bool) -> dict:
        with self.lock:
            self.auto_compounding = bool(enabled)
            self.save_state()
            logger.info(f"Updated Paper Trader compounding engine: {self.auto_compounding}")
            return {"status": "success", "auto_compounding": self.auto_compounding}

    def set_milestone_mode(self, enabled: bool) -> dict:
        with self.lock:
            self.milestone_mode = bool(enabled)
            self.save_state()
            logger.info(f"Updated Paper Trader 10x Milestone Compounding mode: {self.milestone_mode}")
            return {"status": "success", "milestone_mode": self.milestone_mode}

    def set_scanner_mode(self, enabled: bool) -> dict:
        with self.lock:
            self.scanner_mode = bool(enabled)
            self.save_state()
            logger.info(f"Updated Paper Trader scanner_mode to {self.scanner_mode}")
            return {"status": "success", "scanner_mode": self.scanner_mode}

    def set_scanner_universe(self, symbols: List[str]) -> dict:
        with self.lock:
            cleaned = [s.strip().upper() for s in symbols if s.strip()]
            if cleaned:
                self.scanner_universe = cleaned
                self.save_state()
            return {"status": "success", "scanner_universe": self.scanner_universe}

    def get_scanner_radar(self) -> dict:
        now = time.time()
        # Refresh radar if empty or older than 8 seconds
        if not self.radar_scan_results or (now - getattr(self, "last_radar_scan_time", 0.0)) > 8.0:
            radar = self.scan_market_universe()
        else:
            with self.lock:
                radar = list(self.radar_scan_results)
        return {
            "scanner_mode": getattr(self, "scanner_mode", True),
            "universe_count": len(getattr(self, "scanner_universe", [])),
            "active_position_symbol": self.position.get("symbol") if self.position else None,
            "radar": radar,
            "timestamp": now
        }

    def get_dual_model_status(self) -> dict:
        now = time.time()
        with self.lock:
            if not self.radar_scan_results or (now - getattr(self, "last_radar_scan_time", 0.0)) > 8.0:
                self.scan_market_universe()
            return {
                "active_symbol": self.position.get("symbol") if self.position else self.symbol,
                "enforce_dual_consensus": dual_model_engine.enforce_dual_consensus,
                "telemetry": getattr(self, "dual_model_telemetry", {}),
                "timestamp": now
            }

    def configure_dual_model(self, enforce_consensus: bool) -> dict:
        dual_model_engine.enforce_dual_consensus = bool(enforce_consensus)
        return {
            "status": "success",
            "enforce_dual_consensus": dual_model_engine.enforce_dual_consensus
        }

    def get_status(self) -> dict:
        with self.lock:
            elapsed_sec = (time.time() - self.started_at) if (self.is_running and self.started_at > 0) else 0
            elapsed_h = int(elapsed_sec // 3600)
            elapsed_m = int((elapsed_sec % 3600) // 60)
            elapsed_s = int(elapsed_sec % 60)

            fee_rate = self.maker_fee_rate if self.use_maker_execution else self.taker_fee_rate
            total_pos_val = 0.0
            total_unrealized_pnl = 0.0
            total_margin_alloc = 0.0
            enriched_positions = []

            for p_sym, pos_obj in self.positions.items():
                side = pos_obj.get("side", "LONG").upper()
                entry = pos_obj["entry_price"]
                units = pos_obj["units"]
                margin = pos_obj.get("margin_allocated", (units * entry) / max(1.0, pos_obj.get("leverage", self.leverage)))
                pos_px = float(pos_obj.get("current_price", entry))
                est_fee = (units * pos_px) * fee_rate

                if side == "LONG":
                    gross_pnl = (pos_px - entry) * units
                else:
                    gross_pnl = (entry - pos_px) * units

                u_pnl = gross_pnl - est_fee
                u_pnl_pct = (u_pnl / max(1.0, margin)) * 100.0
                p_val = max(0.0, margin + u_pnl)
                total_pos_val += p_val
                total_unrealized_pnl += u_pnl
                total_margin_alloc += margin

                en_pos = dict(pos_obj)
                en_pos["unrealized_pnl"] = round(u_pnl, 2)
                en_pos["unrealized_pnl_pct"] = round(u_pnl_pct, 2)
                enriched_positions.append(en_pos)

            total_equity = self.cash + total_pos_val
            net_pnl = total_equity - self.initial_capital
            net_pnl_pct = (net_pnl / max(self.initial_capital, 1e-4)) * 100.0

            closed_trades = [t for t in self.trades if any(k in t.get("side", "") for k in ["SELL", "CLOSE", "STOP_LOSS", "TAKE_PROFIT", "LIQUIDATION"])]
            winning_trades = [t for t in closed_trades if t.get("pnl_usd", 0) > 0]
            win_rate = (len(winning_trades) / len(closed_trades) * 100.0) if closed_trades else 0.0

            start_idx = getattr(self, "trades_at_session_start", 0)
            session_trades = self.trades[start_idx:]
            closed_session_trades = [t for t in session_trades if any(k in t.get("side", "") for k in ["SELL", "CLOSE", "STOP_LOSS", "TAKE_PROFIT", "LIQUIDATION"])]
            winning_session_trades = [t for t in closed_session_trades if t.get("pnl_usd", 0) > 0]
            session_wr = (len(winning_session_trades) / max(1, len(closed_session_trades)) * 100.0) if closed_session_trades else 0.0

            # Milestone Ladder calculation
            targets = sorted(self.milestone_targets)
            next_target = targets[-1]
            prev_target = self.session_start_capital
            for t in targets:
                if total_equity < t:
                    next_target = t
                    break
                prev_target = t

            milestone_range = max(10.0, next_target - prev_target)
            progress_val = max(0.0, total_equity - prev_target)
            milestone_progress_pct = min(100.0, round((progress_val / milestone_range) * 100.0, 1))
            compound_multiplier = round(total_equity / max(1.0, self.session_start_capital), 2)

            from backend.news_engine import news_engine
            from backend.neural_dream import intelligent_dream_trainer
            from backend.research_engine import research_engine
            news_summary = news_engine.get_aggregate_market_sentiment()

            res_metrics = getattr(self, "latest_research_metrics", None)
            acad_synth = getattr(self, "latest_academic_synthesis", "")
            if not res_metrics:
                m_dummy = {
                    "ticker": self.symbol,
                    "price": self.current_price,
                    "rsi": 50.0,
                    "sma_fast": self.current_price,
                    "sma_slow": self.current_price,
                    "volume_spike": 1.0,
                    "atr_14": self.current_atr,
                    "cvd_delta": 0.0,
                    "funding_rate": self.current_funding_rate,
                    "htf_regime": self.current_htf_regime
                }
                res_metrics = research_engine.compute_research_features(m_dummy, news_summary)
                acad_synth = research_engine.generate_academic_synthesis(res_metrics, m_dummy)

            # Primary position for backward compatibility
            primary_pos = enriched_positions[0] if enriched_positions else None
            first_unrealized = primary_pos.get("unrealized_pnl", 0.0) if primary_pos else 0.0
            first_unrealized_pct = primary_pos.get("unrealized_pnl_pct", 0.0) if primary_pos else 0.0

            return {
                "is_running": self.is_running,
                "strategy_mode": getattr(self, "strategy_mode", "TREND"),
                "trend_params": asdict(self.trend_params) if hasattr(self, "trend_params") else {},
                "session_id": getattr(self, "session_id", ""),
                "symbol": self.symbol,
                "interval": self.interval,
                "timeline_label": getattr(self, "timeline_label", f"{self.interval} Trader"),
                "htf_interval": self.htf_interval,
                "htf_regime": self.current_htf_regime,
                "current_atr": self.current_atr,
                "funding_rate": self.current_funding_rate,
                "funding_skew": self.current_funding_skew,
                "asymmetric_rrr": "2.5:1",
                "execution_mode": "MAKER_POST_ONLY (0.02%)" if self.use_maker_execution else "TAKER_MARKET (0.075%)",
                "started_at": self.started_at,
                "target_duration_hours": self.target_duration_hours,
                "elapsed_formatted": f"{elapsed_h}h {elapsed_m}m {elapsed_s}s",
                "elapsed_hours": round(elapsed_sec / 3600.0, 2),
                "remaining_hours": round(max(0.0, self.target_duration_hours - (elapsed_sec / 3600.0)), 2),
                "initial_capital": round(self.initial_capital, 2),
                "session_start_capital": round(getattr(self, "session_start_capital", self.initial_capital), 2),
                "cash": round(self.cash, 2),
                "total_equity": round(total_equity, 2),
                "monthly_deposit": getattr(self, "monthly_deposit", 0.0),
                "next_deposit_at": (self.last_deposit_ts + self.DEPOSIT_INTERVAL_S) if getattr(self, "monthly_deposit", 0.0) > 0 else None,
                "equity_target": getattr(self, "equity_target", 10000.0),
                "target_progress_pct": round(total_equity / max(getattr(self, "equity_target", 10000.0), 1e-9) * 100.0, 1),
                "net_pnl": round(net_pnl, 2),
                "net_pnl_pct": round(net_pnl_pct, 2),
                "session_trades_count": len(session_trades),
                "session_win_rate": round(session_wr, 1),
                "current_price": round(float(primary_pos.get("current_price", primary_pos.get("entry_price", self.current_price))) if primary_pos else self.current_price, 4 if (float(primary_pos.get("current_price", primary_pos.get("entry_price", self.current_price))) if primary_pos else self.current_price) < 1.0 else 2),
                "position": primary_pos,
                "positions": enriched_positions,
                "max_concurrent_positions": getattr(self, "max_concurrent_positions", 3),
                "active_position_count": len(enriched_positions),
                "fear_and_greed": news_engine.get_fear_and_greed(),
                "unrealized_pnl": round(total_unrealized_pnl, 2),
                "unrealized_pnl_pct": round((total_unrealized_pnl / max(1.0, total_margin_alloc)) * 100.0 if total_margin_alloc > 0 else 0.0, 2),
                "total_maker_fee_savings_usd": round(getattr(self, "total_maker_fee_savings_usd", 0.0), 2),
                "partial_tp_ratio": getattr(self, "partial_tp_ratio", 0.50),
                "tp1_atr_mult": getattr(self, "tp1_atr_mult", 1.2),
                "confidence_gate": self.confidence_gate,
                "sl_atr_mult": self.atr_multiplier_stop,
                "tp_atr_mult": round(self.atr_multiplier_stop * self.target_to_stop_ratio, 2),
                "active_policy_source": getattr(self, "active_policy_source", "NEURAL_MARKET_INTELLIGENCE"),
                "active_formula": getattr(self, "active_formula", "Deep PyTorch 18-D Neural Policy (Bi-Directional Long/Short, 10x Isolated Margin, Compounding Engine)"),
                "is_dream_deployed": True,
                "leverage": self.leverage,
                "auto_compounding": self.auto_compounding,
                "milestone_mode": getattr(self, "milestone_mode", True),
                "consecutive_losses": getattr(self, "consecutive_losses", 0),
                "milestone_phase_label": "Phase 1: Scrappy Growth (< €1,500)" if total_equity < 1500.0 else ("Phase 2: Capital Expansion (€1,500 - €3,500)" if total_equity < 3500.0 else "Phase 3: Milestone Lock (€3,500 - €5,000+)"),
                "max_equity_risk_pct": self.max_equity_risk_pct,
                "milestone_ladder": targets,
                "milestone_current_target": next_target,
                "milestone_prev_target": prev_target,
                "milestone_progress_pct": milestone_progress_pct,
                "compound_multiplier": compound_multiplier,
                "news_sentiment": news_summary.get("score", 0.25),
                "news_sentiment_label": news_summary.get("label", "Mildly Bullish"),
                "top_news_headline": news_summary.get("top_headline", ""),
                "neural_factor_weights": intelligent_dream_trainer.learned_factor_weights,
                "neural_metadata": intelligent_dream_trainer.model_metadata,
                "research_metrics": res_metrics,
                "academic_synthesis": acad_synth,
                "active_model_type": getattr(self, "active_policy_source", "NEURAL_MARKET_INTELLIGENCE"),
                "total_trades": len(self.trades),
                "closed_trades_count": len(closed_trades),
                "win_rate": round(win_rate, 1),
                "recent_trades": self.trades[-30:],
                "recent_snapshots": self.hourly_snapshots[-60:],
                "recent_audits": self.audit_log[-20:],
                "scanner_mode": getattr(self, "scanner_mode", True),
                "scanner_universe": getattr(self, "scanner_universe", []),
                "radar_scan_results": getattr(self, "radar_scan_results", []),
                "dual_model_telemetry": getattr(self, "dual_model_telemetry", {}),
                "active_trade_symbol": primary_pos.get("symbol") if primary_pos else None
            }

paper_trader = PaperTrader()
