"""
5-Minute Futures Zero-Hindsight Walk-Forward Simulation Engine
============================================================
High-speed vectorized 5m futures walk-forward engine for BTCUSDT.
- Zero lookahead: Indicator values at index t are strictly causal functions of past prices [0 ... t].
- Real 5m crypto futures data (17,000+ bars across ~60 continuous days).
- Isolated margin futures simulation with 10x leverage.
- Symmetric Longs and Shorts with regime filters (blocking toxic chop).
- Quarter-Kelly dynamic compounding with strict <= 2.5% equity risk cap.
- Trailing ratchet profit locks (+1.5R activates ratchet, locking +0.8R minimum).
- Rolling online GPU retraining every 500 bars.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("WalkForward5mFutures")

# ---------------------------------------------------------
# Neural Policy Network for 5m Futures
# ---------------------------------------------------------
class Futures5mPolicyNet(nn.Module):
    def __init__(self, input_dim: int = 18):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU()
        )
        self.action_head = nn.Linear(64, 3)  # 0: BUY (Long), 1: SELL (Short), 2: HOLD
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.risk_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 2),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        logits = self.action_head(h)
        conf = self.confidence_head(h)
        risks = self.risk_head(h) + 1.0
        return logits, conf, risks


# ---------------------------------------------------------
# High-Speed Vectorized Causal Indicator Extraction
# ---------------------------------------------------------
def precompute_vectorized_indicators(candles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    """
    Computes all causal features across the entire historical series in a single vectorized pass.
    Strictly causal: Indicator at index t depends strictly on data points 0 to t.
    """
    n = len(candles)
    closes = pd.Series([c["close"] for c in candles], dtype=np.float64)
    highs = pd.Series([c["high"] for c in candles], dtype=np.float64)
    lows = pd.Series([c["low"] for c in candles], dtype=np.float64)
    volumes = pd.Series([c["volume"] for c in candles], dtype=np.float64)

    # 1. EMAs & Spread
    ema_fast = closes.ewm(span=9, adjust=False).mean()
    ema_slow = closes.ewm(span=21, adjust=False).mean()
    ema_htf = closes.ewm(span=50, adjust=False).mean()
    spread_pct = ((ema_fast - ema_slow) / ema_slow.replace(0, 1e-6)) * 100.0

    # 2. ATR 14
    tr1 = highs - lows
    tr2 = (highs - closes.shift(1)).abs()
    tr3 = (lows - closes.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=1).mean()
    atr_exp = atr_14 / tr.rolling(window=7, min_periods=1).mean().replace(0, 1e-6)

    # 3. RSI 14
    diffs = closes.diff()
    gains = diffs.clip(lower=0).rolling(window=14, min_periods=1).mean()
    losses = (-diffs.clip(upper=0)).rolling(window=14, min_periods=1).mean().replace(0, 1e-6)
    rs = gains / losses
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)

    # 4. Volume Spike & CVD Momentum
    avg_vol = volumes.rolling(window=20, min_periods=1).mean().replace(0, 1e-4)
    vol_spike = volumes / avg_vol

    ranges = (highs - lows).replace(0, 1e-6)
    pos_in_range = (closes - lows) / ranges
    cvd_deltas = (pos_in_range - 0.5) * 2.0 * volumes
    cvd_mom = cvd_deltas.rolling(window=5, min_periods=1).sum()

    # 5. HTF 4h Regime Filter (48 bars = 4 hours)
    sma_4h = closes.rolling(window=48, min_periods=1).mean()

    feature_dicts = []
    tensor_rows = []

    for i in range(n):
        px = float(closes.iloc[i])
        ef = float(ema_fast.iloc[i])
        es = float(ema_slow.iloc[i])
        eh = float(ema_htf.iloc[i])
        s_pct = float(spread_pct.iloc[i])
        atr = float(atr_14.iloc[i])
        exp = float(atr_exp.iloc[i])
        r = float(rsi.iloc[i])
        v_spk = float(vol_spike.iloc[i])
        cvd = float(cvd_mom.iloc[i])
        s_4h = float(sma_4h.iloc[i])

        if exp > 1.7 and abs(s_pct) < 0.15:
            regime = "TOXIC_CHOP"
            reg_code = 0.0
        elif px > s_4h and s_pct > 0.15:
            regime = "TREND_BULL"
            reg_code = 1.0
        elif px < s_4h and s_pct < -0.15:
            regime = "TREND_BEAR"
            reg_code = -1.0
        else:
            regime = "RANGE_BOUND"
            reg_code = 0.0

        # Synchronized intraday news/catalyst simulation strictly based on past 12-bar return
        past_12 = (px - closes.iloc[max(0, i - 12)]) / max(1.0, closes.iloc[max(0, i - 12)])
        seed = i % 1000000
        shock = (seed % 150 == 0)
        news_score = float(np.clip(past_12 * 8.0 + ((seed % 19) - 9) * 0.02, -0.9, 0.9))
        if shock:
            news_score = float(np.sign(news_score or 1.0) * 0.85)

        f1_rsi = (r - 50.0) / 50.0
        f2_spread = s_pct * 0.5
        f3_htf_dist = ((px - eh) / max(1e-6, eh)) * 30.0
        f4_vol = np.clip(v_spk / 3.0, 0.2, 2.0)
        f5_cvd = np.clip(cvd / 1000.0, -1.0, 1.0)
        f6_atr = min(0.05, atr / px) / 0.05
        f7_exp = min(3.0, exp) / 3.0
        f8_reg = reg_code
        f9_news = news_score
        f10_shock = 1.0 if shock else 0.0
        f11_cat = 0.5 if shock else 0.0
        f12_inter = news_score * (1.0 if s_pct > 0 else -1.0)
        f13_loss = 0.5 + 0.3 * np.tanh(f1_rsi)
        f14_vpin = float(np.clip(v_spk * 0.2 + abs(f5_cvd) * 0.3, 0.1, 0.95))
        f15_refl = float(np.clip(f2_spread * news_score, -1.0, 1.0))
        f16_crowd = float(-1.0 * f1_rsi if abs(f1_rsi) > 0.6 else 0.0)
        f17_chop = 1.0 if regime == "TOXIC_CHOP" else 0.0
        f18_fund = 0.1

        feats = [
            f1_rsi, f2_spread, f3_htf_dist, f4_vol, f5_cvd, f6_atr, f7_exp,
            f8_reg, f9_news, f10_shock, f11_cat, f12_inter,
            f13_loss, f14_vpin, f15_refl, f16_crowd, f17_chop, f18_fund
        ]

        feature_dicts.append({
            "price": px,
            "rsi": round(r, 2),
            "atr_14": round(atr, 2),
            "spread_pct": round(s_pct, 3),
            "htf_regime": regime,
            "news_score": round(news_score, 3)
        })
        tensor_rows.append(feats)

    features_tensor = torch.tensor(tensor_rows, dtype=torch.float32)
    return feature_dicts, features_tensor


# ---------------------------------------------------------
# 5-Minute Futures Walk-Forward Simulator
# ---------------------------------------------------------
class Futures5mSimulator:
    def __init__(
        self,
        candles: List[Dict[str, Any]],
        initial_capital: float = 300.0,
        leverage: float = 10.0,
        auto_compounding: bool = True
    ):
        self.candles = candles
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.leverage = leverage
        self.auto_compounding = auto_compounding

        self.max_equity_risk_pct = 0.025  # 2.5% max equity risk budget
        self.maintenance_margin_rate = 0.005  # 0.5% MMR
        self.fee_rate = 0.0004  # 0.04% maker/taker institutional futures fee
        self.slippage_pct = 0.0002  # 2 bps slippage

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Futures5mPolicyNet(input_dim=18).to(self.device)

        logger.info("Precomputing vectorized causal indicators across dataset...")
        self.feature_dicts, self.features_tensor = precompute_vectorized_indicators(candles)
        self.features_tensor = self.features_tensor.to(self.device)
        logger.info(f"Causal features ready for {len(candles)} 5m bars.")

        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.position = None
        self.last_trade_closed_bar = -100
        self.cooldown_bars = 6  # 30 mins cooldown
        self.consecutive_losses = 0

        self.milestones = [500.0, 1000.0, 2500.0, 5000.0, 10000.0, 25000.0]
        self.reached_milestones = set()

    def train_policy_on_window(self, end_idx: int, epochs: int = 12):
        """Train neural policy strictly on slice [0 ... end_idx] using future returns."""
        if end_idx < 100:
            return

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion_action = nn.CrossEntropyLoss()
        criterion_risk = nn.MSELoss()

        start_idx = max(30, end_idx - 1500)
        indices = list(range(start_idx, end_idx - 6))
        if not indices:
            return

        closes = [self.candles[k]["close"] for k in range(end_idx)]
        action_targets = []
        risk_targets = []

        for i in indices:
            atr = max(self.feature_dicts[i]["atr_14"], 1.0)
            ret_atr = (closes[min(end_idx - 1, i + 6)] - closes[i]) / atr
            ns = self.feature_dicts[i]["news_score"]

            if ret_atr > 0.8 and ns > -0.3:
                act = 0  # BUY
                sl, tp = 1.8, 3.0
            elif ret_atr < -0.8 and ns < 0.3:
                act = 1  # SELL
                sl, tp = 1.8, 3.0
            else:
                act = 2  # HOLD
                sl, tp = 2.0, 2.5
            action_targets.append(act)
            risk_targets.append([sl, tp])

        X = self.features_tensor[indices]
        Y_act = torch.tensor(action_targets, dtype=torch.long, device=self.device)
        Y_risk = torch.tensor(risk_targets, dtype=torch.float32, device=self.device)

        # Class balancing weights
        class_weights = torch.tensor([2.2, 2.2, 0.7], device=self.device)
        criterion_action = nn.CrossEntropyLoss(weight=class_weights)

        for _ in range(epochs):
            optimizer.zero_grad()
            logits, _, risks = self.model(X)
            loss = criterion_action(logits, Y_act) + 0.2 * criterion_risk(risks, Y_risk)
            loss.backward()
            optimizer.step()

        self.model.eval()

    def infer_at_step(self, t: int) -> Tuple[str, float]:
        with torch.no_grad():
            x = self.features_tensor[t].unsqueeze(0)
            logits, _, _ = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            p_buy, p_sell, p_hold = float(probs[0]), float(probs[1]), float(probs[2])

        edge = 0.12
        if p_buy >= 0.40 and p_buy > (p_sell + edge):
            return "buy", p_buy
        elif p_sell >= 0.40 and p_sell > (p_buy + edge):
            return "sell", p_sell
        else:
            return "hold", p_hold

    def compute_quarter_kelly_margin(self, equity: float, stop_dist_pct: float) -> float:
        closed = [t for t in self.trades if "CLOSE" in t["side"]]
        wins = [t for t in closed if t["pnl_usd"] > 0]
        p = (len(wins) / len(closed)) if len(closed) >= 5 else 0.58
        p = float(np.clip(p, 0.45, 0.75))
        b = 2.0
        kelly = (p * (b + 1.0) - 1.0) / b
        quarter_kelly = max(0.008, kelly * 0.25)
        risk_pct = min(quarter_kelly, self.max_equity_risk_pct)

        risk_cap = equity * risk_pct
        safe_stop = max(stop_dist_pct, 0.004)
        notional = risk_cap / safe_stop
        req_margin = notional / max(1.0, self.leverage)
        return float(min(req_margin, self.cash * 0.85))

    def run_simulation(self, warmup_bars: int = 1500, retrain_freq: int = 500):
        total_bars = len(self.candles)
        logger.info(f"Initiating 5m Futures Walk-Forward across {total_bars} bars on {self.device}...")
        logger.info(f"Warmup in-sample period: {warmup_bars} bars (~5.2 days) | Retrain cadence: every {retrain_freq} bars (~1.7 days)")
        logger.info(f"Initial Capital: ${self.initial_capital:.2f} | Isolated Leverage: {self.leverage:.0f}x | Compounding: {self.auto_compounding}")

        # Warmup training
        self.train_policy_on_window(warmup_bars, epochs=16)

        for t in range(warmup_bars, total_bars):
            # Rolling retraining
            if (t - warmup_bars) > 0 and (t - warmup_bars) % retrain_freq == 0:
                self.train_policy_on_window(t, epochs=8)

            cur_candle = self.candles[t]
            ts_str = cur_candle.get("timestamp", f"Bar_{t}")
            m = self.feature_dicts[t]

            # ---------------------------------------------------------
            # Active Position Management (5m Futures)
            # ---------------------------------------------------------
            if self.position is not None:
                pos = self.position
                side = pos["side"]
                entry_px = pos["entry_price"]
                units = pos["units"]
                margin = pos["margin_allocated"]
                stop_dist = pos["stop_dist"]

                if side == "LONG":
                    pos["highest_price"] = max(pos["highest_price"], cur_candle["high"])

                    # Liquidation Check
                    if cur_candle["low"] <= pos["liquidation_price"]:
                        loss = margin
                        self.trades.append({
                            "id": len(self.trades) + 1,
                            "bar": t,
                            "timestamp": ts_str,
                            "side": "LIQUIDATION_CLOSE_LONG",
                            "entry": round(entry_px, 2),
                            "exit": round(pos["liquidation_price"], 2),
                            "units": round(units, 6),
                            "pnl_usd": round(-loss, 2),
                            "pnl_pct": -100.0,
                            "capital": round(self.cash, 2),
                            "reason": f"Isolated Margin Liquidation ({pos['leverage']}x)"
                        })
                        self.last_trade_closed_bar = t
                        self.position = None

                    elif self.position is not None:
                        # Trailing Ratchet: When high >= entry + 1.5 * stop_dist (+1.5R)
                        if cur_candle["high"] >= pos["breakeven_trigger"]:
                            pos["is_trailing"] = True
                            pos["stop_loss_price"] = max(pos["stop_loss_price"], entry_px + (0.8 * stop_dist))
                            trail = pos["highest_price"] - (1.0 * stop_dist)
                            if trail > pos["stop_loss_price"]:
                                pos["stop_loss_price"] = trail

                        # Take Profit Target Hit (+2.0R)
                        if cur_candle["high"] >= pos["take_profit_price"]:
                            exit_px = pos["take_profit_price"]
                            gross = (exit_px - entry_px) * units
                            fee = (units * exit_px) * self.fee_rate
                            pnl = gross - fee
                            self.cash += (margin + pnl)
                            self.trades.append({
                                "id": len(self.trades) + 1,
                                "bar": t,
                                "timestamp": ts_str,
                                "side": "TAKE_PROFIT_CLOSE_LONG",
                                "entry": round(entry_px, 2),
                                "exit": round(exit_px, 2),
                                "units": round(units, 6),
                                "pnl_usd": round(pnl, 2),
                                "pnl_pct": round((pnl / margin) * 100.0, 2),
                                "capital": round(self.cash, 2),
                                "reason": f"Asymmetric Target Hit (+2.0R, {pos['leverage']}x)"
                            })
                            self.last_trade_closed_bar = t
                            self.last_trade_was_loss = False
                            self.consecutive_losses = 0
                            self.position = None

                        # Stop Loss or Trailing Ratchet Exit
                        elif cur_candle["low"] <= pos["stop_loss_price"]:
                            exit_px = pos["stop_loss_price"]
                            gross = (exit_px - entry_px) * units
                            fee = (units * exit_px) * self.fee_rate
                            pnl = gross - fee
                            self.cash += max(0.0, margin + pnl)
                            is_trail_profit = pos.get("is_trailing", False) and pnl > 0
                            side_label = "TRAILING_RATCHET_CLOSE_LONG" if is_trail_profit else "STOP_LOSS_CLOSE_LONG"
                            self.trades.append({
                                "id": len(self.trades) + 1,
                                "bar": t,
                                "timestamp": ts_str,
                                "side": side_label,
                                "entry": round(entry_px, 2),
                                "exit": round(exit_px, 2),
                                "units": round(units, 6),
                                "pnl_usd": round(pnl, 2),
                                "pnl_pct": round((pnl / margin) * 100.0, 2),
                                "capital": round(self.cash, 2),
                                "reason": f"{'Trailing Ratchet Lock' if is_trail_profit else 'Dynamic 5m ATR Stop'} ({pos['leverage']}x)"
                            })
                            self.last_trade_closed_bar = t
                            self.last_trade_was_loss = (pnl < 0)
                            if pnl < 0:
                                self.consecutive_losses += 1
                            else:
                                self.consecutive_losses = 0
                            self.position = None

                else:  # SHORT
                    pos["lowest_price"] = min(pos["lowest_price"], cur_candle["low"])

                    # Liquidation Check
                    if cur_candle["high"] >= pos["liquidation_price"]:
                        loss = margin
                        self.trades.append({
                            "id": len(self.trades) + 1,
                            "bar": t,
                            "timestamp": ts_str,
                            "side": "LIQUIDATION_CLOSE_SHORT",
                            "entry": round(entry_px, 2),
                            "exit": round(pos["liquidation_price"], 2),
                            "units": round(units, 6),
                            "pnl_usd": round(-loss, 2),
                            "pnl_pct": -100.0,
                            "capital": round(self.cash, 2),
                            "reason": f"Isolated Margin Liquidation ({pos['leverage']}x)"
                        })
                        self.last_trade_closed_bar = t
                        self.position = None

                    elif self.position is not None:
                        # Trailing Ratchet: When low <= entry - 1.5 * stop_dist (+1.5R)
                        if cur_candle["low"] <= pos["breakeven_trigger"]:
                            pos["is_trailing"] = True
                            pos["stop_loss_price"] = min(pos["stop_loss_price"], entry_px - (0.8 * stop_dist))
                            trail = pos["lowest_price"] + (1.0 * stop_dist)
                            if trail < pos["stop_loss_price"]:
                                pos["stop_loss_price"] = trail

                        # Take Profit Target Hit (+2.0R)
                        if cur_candle["low"] <= pos["take_profit_price"]:
                            exit_px = pos["take_profit_price"]
                            gross = (entry_px - exit_px) * units
                            fee = (units * exit_px) * self.fee_rate
                            pnl = gross - fee
                            self.cash += (margin + pnl)
                            self.trades.append({
                                "id": len(self.trades) + 1,
                                "bar": t,
                                "timestamp": ts_str,
                                "side": "TAKE_PROFIT_CLOSE_SHORT",
                                "entry": round(entry_px, 2),
                                "exit": round(exit_px, 2),
                                "units": round(units, 6),
                                "pnl_usd": round(pnl, 2),
                                "pnl_pct": round((pnl / margin) * 100.0, 2),
                                "capital": round(self.cash, 2),
                                "reason": f"Asymmetric Target Hit (+2.0R, {pos['leverage']}x)"
                            })
                            self.last_trade_closed_bar = t
                            self.last_trade_was_loss = False
                            self.consecutive_losses = 0
                            self.position = None

                        # Stop Loss or Trailing Ratchet Exit
                        elif cur_candle["high"] >= pos["stop_loss_price"]:
                            exit_px = pos["stop_loss_price"]
                            gross = (entry_px - exit_px) * units
                            fee = (units * exit_px) * self.fee_rate
                            pnl = gross - fee
                            self.cash += max(0.0, margin + pnl)
                            is_trail_profit = pos.get("is_trailing", False) and pnl > 0
                            side_label = "TRAILING_RATCHET_CLOSE_SHORT" if is_trail_profit else "STOP_LOSS_CLOSE_SHORT"
                            self.trades.append({
                                "id": len(self.trades) + 1,
                                "bar": t,
                                "timestamp": ts_str,
                                "side": side_label,
                                "entry": round(entry_px, 2),
                                "exit": round(exit_px, 2),
                                "units": round(units, 6),
                                "pnl_usd": round(pnl, 2),
                                "pnl_pct": round((pnl / margin) * 100.0, 2),
                                "capital": round(self.cash, 2),
                                "reason": f"{'Trailing Ratchet Lock' if is_trail_profit else 'Dynamic 5m ATR Stop'} ({pos['leverage']}x)"
                            })
                            self.last_trade_closed_bar = t
                            self.last_trade_was_loss = (pnl < 0)
                            if pnl < 0:
                                self.consecutive_losses += 1
                            else:
                                self.consecutive_losses = 0
                            self.position = None

                # Check milestones
                for m_val in self.milestones:
                    if self.cash >= m_val and m_val not in self.reached_milestones:
                        self.reached_milestones.add(m_val)
                        logger.info(f"*** MILESTONE HIT *** Account equity crossed ${m_val:,.2f} at bar {t} ({ts_str})!")

            # ---------------------------------------------------------
            # Signal Evaluation & Order Entry (5m Futures)
            # ---------------------------------------------------------
            cons_losses = getattr(self, "consecutive_losses", 0)
            if cons_losses >= 2:
                required_cooldown = 72  # 6 hours circuit breaker after 2 losses
            elif cons_losses == 1:
                required_cooldown = 24  # 2 hours after 1 loss
            else:
                required_cooldown = 6   # 30 mins after a winning trade

            in_cooldown = (t - self.last_trade_closed_bar) < required_cooldown
            if self.position is None and not in_cooldown:
                choice, conf = self.infer_at_step(t)
                cur_px = cur_candle["close"]
                atr = m["atr_14"]
                stop_dist = max(atr * 2.2, cur_px * 0.008)
                stop_dist_pct = stop_dist / cur_px

                if choice in ["buy", "sell"] and conf >= 0.40:
                    margin_alloc = self.compute_quarter_kelly_margin(self.cash, stop_dist_pct)
                    # Throttle risk by 50% if recovering from a recent loss
                    if cons_losses >= 1:
                        margin_alloc *= 0.60
                    if margin_alloc >= 15.0 and self.cash >= margin_alloc:
                        fee = margin_alloc * self.fee_rate
                        eff_margin = margin_alloc - fee
                        notional = eff_margin * self.leverage

                        is_bull_ok = (m["htf_regime"] == "TREND_BULL")
                        is_bear_ok = (m["htf_regime"] == "TREND_BEAR")

                        if choice == "buy" and is_bull_ok:
                            exec_px = cur_px * (1.0 + self.slippage_pct)
                            units = notional / exec_px
                            sl_px = exec_px - stop_dist
                            tp_px = exec_px + (2.0 * stop_dist)
                            liq_px = exec_px * (1.0 - (1.0 / self.leverage) + self.maintenance_margin_rate)

                            self.cash -= margin_alloc
                            self.position = {
                                "side": "LONG",
                                "entry_price": exec_px,
                                "units": units,
                                "leverage": self.leverage,
                                "margin_allocated": margin_alloc,
                                "stop_loss_price": sl_px,
                                "take_profit_price": tp_px,
                                "breakeven_trigger": exec_px + (1.5 * stop_dist),
                                "highest_price": exec_px,
                                "lowest_price": exec_px,
                                "liquidation_price": liq_px,
                                "atr_14": atr,
                                "stop_dist": stop_dist,
                                "is_trailing": False
                            }

                        elif choice == "sell" and is_bear_ok:
                            exec_px = cur_px * (1.0 - self.slippage_pct)
                            units = notional / exec_px
                            sl_px = exec_px + stop_dist
                            tp_px = exec_px - (2.0 * stop_dist)
                            liq_px = exec_px * (1.0 + (1.0 / self.leverage) - self.maintenance_margin_rate)

                            self.cash -= margin_alloc
                            self.position = {
                                "side": "SHORT",
                                "entry_price": exec_px,
                                "units": units,
                                "leverage": self.leverage,
                                "margin_allocated": margin_alloc,
                                "stop_loss_price": sl_px,
                                "take_profit_price": tp_px,
                                "breakeven_trigger": exec_px - (1.5 * stop_dist),
                                "highest_price": exec_px,
                                "lowest_price": exec_px,
                                "liquidation_price": liq_px,
                                "atr_14": atr,
                                "stop_dist": stop_dist,
                                "is_trailing": False
                            }

            # Record equity
            unrealized = 0.0
            if self.position is not None:
                p = self.position
                px = cur_candle["close"]
                if p["side"] == "LONG":
                    unrealized = (px - p["entry_price"]) * p["units"]
                else:
                    unrealized = (p["entry_price"] - px) * p["units"]
                cur_equity = self.cash + p["margin_allocated"] + unrealized
            else:
                cur_equity = self.cash

            self.equity_curve.append({
                "bar": t,
                "timestamp": ts_str,
                "equity": round(cur_equity, 2),
                "in_trade": self.position is not None
            })

        # Final cleanup
        if self.position is not None:
            last_c = self.candles[-1]
            pos = self.position
            px = last_c["close"]
            if pos["side"] == "LONG":
                pnl = (px - pos["entry_price"]) * pos["units"]
            else:
                pnl = (pos["entry_price"] - px) * pos["units"]
            self.cash += max(0.0, pos["margin_allocated"] + pnl)
            self.trades.append({
                "id": len(self.trades) + 1,
                "bar": len(self.candles) - 1,
                "timestamp": last_c.get("timestamp", ""),
                "side": f"END_OF_TEST_CLOSE_{pos['side']}",
                "entry": round(pos["entry_price"], 2),
                "exit": round(px, 2),
                "units": round(pos["units"], 6),
                "pnl_usd": round(pnl, 2),
                "pnl_pct": round((pnl / pos["margin_allocated"]) * 100.0, 2),
                "capital": round(self.cash, 2),
                "reason": "Horizon End"
            })
            self.position = None

        return self._generate_report(warmup_bars)

    def _generate_report(self, warmup_bars: int) -> Dict[str, Any]:
        closed = [t for t in self.trades if "CLOSE" in t["side"]]
        wins = [t for t in closed if t["pnl_usd"] > 0]
        losses = [t for t in closed if t["pnl_usd"] < 0]
        longs = [t for t in closed if "LONG" in t["side"]]
        shorts = [t for t in closed if "SHORT" in t["side"]]

        long_wins = [t for t in longs if t["pnl_usd"] > 0]
        short_wins = [t for t in shorts if t["pnl_usd"] > 0]

        total_gain = sum(t["pnl_usd"] for t in wins)
        total_loss = abs(sum(t["pnl_usd"] for t in losses))
        profit_factor = round(total_gain / max(total_loss, 1e-6), 2)

        win_rate = round((len(wins) / len(closed) * 100.0), 1) if closed else 0.0
        long_win_rate = round((len(long_wins) / len(longs) * 100.0), 1) if longs else 0.0
        short_win_rate = round((len(short_wins) / len(shorts) * 100.0), 1) if shorts else 0.0

        equities = np.array([e["equity"] for e in self.equity_curve])
        peak = np.maximum.accumulate(equities)
        dd = (equities - peak) / peak
        max_dd_pct = round(float(np.min(dd)) * 100.0, 2)
        max_dd_usd = round(float(np.min(equities - peak)), 2)

        rets = np.diff(equities) / np.maximum(equities[:-1], 1e-6)
        if len(rets) > 1 and np.std(rets) > 1e-8:
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(105120))
            downside = rets[rets < 0]
            sortino = float(np.mean(rets) / np.std(downside) * np.sqrt(105120)) if len(downside) > 1 else sharpe
        else:
            sharpe, sortino = 0.0, 0.0

        ending_equity = round(self.cash, 2)
        tot_return_pct = round(((ending_equity - self.initial_capital) / self.initial_capital) * 100.0, 2)

        report = {
            "timeframe": "5m",
            "total_bars": len(self.candles),
            "oos_bars": len(self.candles) - warmup_bars,
            "start_time": self.candles[warmup_bars].get("timestamp", ""),
            "end_time": self.candles[-1].get("timestamp", ""),
            "initial_capital": self.initial_capital,
            "ending_equity": ending_equity,
            "total_return_pct": tot_return_pct,
            "multiplier": round(ending_equity / self.initial_capital, 2),
            "annualized_sharpe": round(sharpe, 2),
            "annualized_sortino": round(sortino, 2),
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_usd": max_dd_usd,
            "total_trades": len(closed),
            "win_rate_pct": win_rate,
            "long_trades": len(longs),
            "long_win_rate_pct": long_win_rate,
            "short_trades": len(shorts),
            "short_win_rate_pct": short_win_rate,
            "profit_factor": profit_factor,
            "milestones_reached": sorted(list(self.reached_milestones))
        }

        print("\n" + "=" * 70)
        print("5-MINUTE FUTURES ZERO-HINDSIGHT WALK-FORWARD SIMULATION RESULTS")
        print("=" * 70)
        print(f"Timeframe:              5-Minute (5m) BTCUSDT Futures")
        print(f"Horizon:                {report['start_time']} to {report['end_time']} ({report['oos_bars']} bars, ~{report['oos_bars']/288:.1f} days)")
        print(f"Initial Capital:        ${self.initial_capital:.2f}")
        print(f"Ending Equity:          ${ending_equity:.2f} ({report['multiplier']}x)")
        print(f"Total Return:           {'+' if tot_return_pct >= 0 else ''}{tot_return_pct}%")
        print(f"Annualized Sharpe:      {report['annualized_sharpe']}")
        print(f"Annualized Sortino:     {report['annualized_sortino']}")
        print(f"Max Drawdown:           {max_dd_pct}% (${max_dd_usd:.2f})")
        print(f"Total Trades:           {len(closed)}")
        print(f"Overall Win Rate:       {win_rate}% (Longs: {long_win_rate}%, Shorts: {short_win_rate}%)")
        print(f"Profit Factor:          {profit_factor}")
        print(f"Milestones Reached:     {report['milestones_reached']}")
        print("=" * 70 + "\n")

        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)

        trades_df = pd.DataFrame(self.trades)
        trades_csv_path = os.path.join(data_dir, "walk_forward_5m_trades.csv")
        trades_df.to_csv(trades_csv_path, index=False)
        logger.info(f"Saved 5m trade blotter to {trades_csv_path}")

        results_json_path = os.path.join(data_dir, "walk_forward_5m_results.json")
        with open(results_json_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved 5m simulation report to {results_json_path}")

        return report


def main():
    logger.info("Downloading 60 days of 5-minute historical candle data for BTC-USD...")
    df = yf.download(tickers="BTC-USD", interval="5m", period="60d", progress=False)

    if df.empty or len(df) < 500:
        logger.error("Failed to download 5m historical data.")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    candles = []
    for idx, row in df.iterrows():
        candles.append({
            "timestamp": idx.strftime("%Y-%m-%d %H:%M"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"])
        })

    logger.info(f"Successfully downloaded {len(candles)} real 5-minute candles ({candles[0]['timestamp']} to {candles[-1]['timestamp']})")

    sim = Futures5mSimulator(
        candles=candles,
        initial_capital=300.0,
        leverage=10.0,
        auto_compounding=True
    )
    sim.run_simulation(warmup_bars=1500, retrain_freq=500)


if __name__ == "__main__":
    main()
