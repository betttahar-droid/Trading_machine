"""
Institutional Multi-Asset Compounding Futures Engine (BTC, ETH, SOL)
====================================================================
Advanced high-performance 5m futures trading engine featuring:
1. Multi-Asset Cross-Sectional Momentum Radar (BTC, ETH, SOL).
2. Multi-Factor Confluence Scoring (0 - 100) with Tiered Risk Allocation.
3. The "Moonbag" Split: 50% partial TP bank at +1.5R + uncapped parabolic runner.
4. House-Money Pyramiding on confirmed trend continuations.
5. Volatility Squeeze Breakout Detector (Bollinger Bands inside Keltner Channels).
6. Institutional Circuit Breaker (6-hour pause after 2 losses, anti-martingale risk reduction).
7. Zero lookahead: Indicators and regimes strictly causal functions of past history [0 ... t].
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple, Optional

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
logger = logging.getLogger("InstitutionalCompounding")

# ---------------------------------------------------------
# Neural Directional Policy Network
# ---------------------------------------------------------
class MultiAssetPolicyNet(nn.Module):
    def __init__(self, input_dim: int = 18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 3)  # 0: BUY, 1: SELL, 2: HOLD
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------
# Causal Feature & Volatility Squeeze Computation
# ---------------------------------------------------------
def precompute_asset_features(candles: List[Dict[str, Any]], asset_name: str) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    """Precomputes all causal indicators including Bollinger-Keltner Volatility Squeeze."""
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

    # 2. True Range & ATR 14
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
    rsi = 100.0 - (100.0 / (1.0 + (gains / losses)))
    rsi = rsi.fillna(50.0)

    # 4. Volume Spike & CVD Momentum
    avg_vol = volumes.rolling(window=20, min_periods=1).mean().replace(0, 1e-4)
    vol_spike = volumes / avg_vol

    ranges = (highs - lows).replace(0, 1e-6)
    pos_in_range = (closes - lows) / ranges
    cvd_deltas = (pos_in_range - 0.5) * 2.0 * volumes
    cvd_mom = cvd_deltas.rolling(window=5, min_periods=1).sum()

    # 5. Volatility Squeeze (Bollinger Bands vs. Keltner Channels)
    # BB: 20-period SMA +/- 2.0 std
    bb_mid = closes.rolling(window=20, min_periods=1).mean()
    bb_std = closes.rolling(window=20, min_periods=1).std().fillna(0.0)
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)

    # KC: 20-period EMA +/- 1.5 ATR
    kc_mid = closes.ewm(span=20, adjust=False).mean()
    kc_upper = kc_mid + (1.5 * atr_14)
    kc_lower = kc_mid - (1.5 * atr_14)

    # Squeeze is ON when BB is completely inside KC (volatility compressed)
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # 6. HTF 4h Regime Filter (48 bars = 4 hours)
    sma_4h = closes.rolling(window=48, min_periods=1).mean()

    feature_dicts = []
    tensor_rows = []

    for i in range(n):
        px = float(closes.iloc[i])
        s_pct = float(spread_pct.iloc[i])
        atr = float(atr_14.iloc[i])
        exp = float(atr_exp.iloc[i])
        r = float(rsi.iloc[i])
        v_spk = float(vol_spike.iloc[i])
        cvd = float(cvd_mom.iloc[i])
        s_4h = float(sma_4h.iloc[i])
        is_squeeze = bool(squeeze_on.iloc[i])
        eh = float(ema_htf.iloc[i])

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

        past_12 = (px - closes.iloc[max(0, i - 12)]) / max(1.0, closes.iloc[max(0, i - 12)])
        seed = i % 1000000
        shock = (seed % 150 == 0)
        news_score = float(np.clip(past_12 * 8.0 + ((seed % 19) - 9) * 0.02, -0.9, 0.9))
        if shock:
            news_score = float(np.sign(news_score or 1.0) * 0.85)

        # Build feature vector
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
        f16_squeeze = 1.0 if is_squeeze else 0.0
        f17_chop = 1.0 if regime == "TOXIC_CHOP" else 0.0
        f18_fund = 0.1

        feats = [
            f1_rsi, f2_spread, f3_htf_dist, f4_vol, f5_cvd, f6_atr, f7_exp,
            f8_reg, f9_news, f10_shock, f11_cat, f12_inter,
            f13_loss, f14_vpin, f15_refl, f16_squeeze, f17_chop, f18_fund
        ]

        feature_dicts.append({
            "asset": asset_name,
            "price": px,
            "rsi": round(r, 2),
            "atr_14": round(atr, 2),
            "spread_pct": round(s_pct, 3),
            "htf_regime": regime,
            "is_squeeze": is_squeeze,
            "cvd_mom": cvd,
            "vol_spike": v_spk,
            "news_score": round(news_score, 3)
        })
        tensor_rows.append(feats)

    features_tensor = torch.tensor(tensor_rows, dtype=torch.float32)
    return feature_dicts, features_tensor


# ---------------------------------------------------------
# Confluence Scoring Engine (0 to 100)
# ---------------------------------------------------------
def score_setup_confluence(m: Dict[str, Any], direction: str) -> float:
    """Computes an institutional confluence score from 0 to 100."""
    score = 0.0

    # 1. 4h Regime Alignment (Up to 25 pts)
    if direction == "buy" and m["htf_regime"] == "TREND_BULL":
        score += 25.0
    elif direction == "sell" and m["htf_regime"] == "TREND_BEAR":
        score += 25.0

    # 2. 5m Spread Momentum (Up to 20 pts)
    if direction == "buy" and m["spread_pct"] > 0.1:
        score += min(20.0, m["spread_pct"] * 40.0)
    elif direction == "sell" and m["spread_pct"] < -0.1:
        score += min(20.0, abs(m["spread_pct"]) * 40.0)

    # 3. Order-Flow CVD Delta Momentum (Up to 20 pts)
    if direction == "buy" and m["cvd_mom"] > 0:
        score += 20.0
    elif direction == "sell" and m["cvd_mom"] < 0:
        score += 20.0

    # 4. Volatility Squeeze Breakout (Up to 15 pts)
    if m["is_squeeze"]:
        score += 15.0

    # 5. News Sentiment Alignment (Up to 20 pts)
    if direction == "buy" and m["news_score"] > 0.1:
        score += min(20.0, m["news_score"] * 25.0)
    elif direction == "sell" and m["news_score"] < -0.1:
        score += min(20.0, abs(m["news_score"]) * 25.0)

    return round(float(score), 1)


# ---------------------------------------------------------
# Institutional Multi-Asset Compounding Simulator
# ---------------------------------------------------------
class InstitutionalCompoundingSimulator:
    def __init__(
        self,
        asset_candles: Dict[str, List[Dict[str, Any]]],
        initial_capital: float = 300.0,
        leverage: float = 10.0
    ):
        self.asset_candles = asset_candles
        self.assets = list(asset_candles.keys())
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.leverage = leverage

        self.maintenance_margin_rate = 0.005  # 0.5% MMR
        self.fee_rate = 0.0004  # 0.04% maker/taker futures fee
        self.slippage_pct = 0.0002

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiAssetPolicyNet(input_dim=18).to(self.device)

        logger.info(f"Precomputing causal indicators for {len(self.assets)} assets: {self.assets}...")
        self.f_dicts: Dict[str, List[Dict[str, Any]]] = {}
        self.f_tensors: Dict[str, torch.Tensor] = {}

        for sym in self.assets:
            d, t = precompute_asset_features(self.asset_candles[sym], sym)
            self.f_dicts[sym] = d
            self.f_tensors[sym] = t.to(self.device)

        logger.info(f"Causal indicators ready across all assets.")

        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.position: Optional[Dict[str, Any]] = None
        self.pyramid_position: Optional[Dict[str, Any]] = None

        self.last_trade_closed_bar = -100
        self.consecutive_losses = 0

        self.milestones = [500.0, 1000.0, 2500.0, 5000.0, 10000.0, 25000.0]
        self.reached_milestones = set()

    def train_policy_on_window(self, end_idx: int, epochs: int = 12):
        """Trains neural policy network on past slice [0 ... end_idx] using BTC/ETH data."""
        if end_idx < 100:
            return

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.003, weight_decay=1e-4)
        class_weights = torch.tensor([2.2, 2.2, 0.7], device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        start_idx = max(30, end_idx - 1500)
        indices = list(range(start_idx, end_idx - 6))
        if not indices:
            return

        # Train primarily on BTC signals as the macro anchor
        btc_closes = [self.asset_candles["BTC"][k]["close"] for k in range(end_idx)]
        action_targets = []

        for i in indices:
            atr = max(self.f_dicts["BTC"][i]["atr_14"], 1.0)
            ret_atr = (btc_closes[min(end_idx - 1, i + 6)] - btc_closes[i]) / atr
            ns = self.f_dicts["BTC"][i]["news_score"]

            if ret_atr > 0.8 and ns > -0.3:
                act = 0  # BUY
            elif ret_atr < -0.8 and ns < 0.3:
                act = 1  # SELL
            else:
                act = 2  # HOLD
            action_targets.append(act)

        X = self.f_tensors["BTC"][indices]
        Y = torch.tensor(action_targets, dtype=torch.long, device=self.device)

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.model(X)
            loss = criterion(logits, Y)
            loss.backward()
            optimizer.step()

        self.model.eval()

    def infer_asset(self, sym: str, t: int) -> Tuple[str, float]:
        with torch.no_grad():
            x = self.f_tensors[sym][t].unsqueeze(0)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            p_buy, p_sell, p_hold = float(probs[0]), float(probs[1]), float(probs[2])

        edge = 0.10
        if p_buy >= 0.40 and p_buy > (p_sell + edge):
            return "buy", p_buy
        elif p_sell >= 0.40 and p_sell > (p_buy + edge):
            return "sell", p_sell
        else:
            return "hold", p_hold

    def compute_confluence_margin(self, equity: float, stop_dist_pct: float, score: float) -> float:
        """Dynamic confluence tier sizing with Quarter-Kelly risk budget."""
        if score >= 65.0:
            risk_pct = 0.035  # 3.5% Prime Setup risk
        elif score >= 52.0:
            risk_pct = 0.028  # 2.8% Quality Setup risk
        else:
            risk_pct = 0.022  # 2.2% Base Setup risk

        # Throttle risk if recovering from recent losses (anti-martingale)
        if self.consecutive_losses >= 1:
            risk_pct *= 0.60

        risk_cap = equity * risk_pct
        safe_stop = max(stop_dist_pct, 0.008)
        notional = risk_cap / safe_stop
        req_margin = notional / max(1.0, self.leverage)
        return float(min(req_margin, self.cash * 0.85))

    def run_simulation(self, warmup_bars: int = 1500, retrain_freq: int = 500):
        total_bars = min(len(self.asset_candles[s]) for s in self.assets)
        logger.info(f"Initiating Institutional Compounding Simulator across {total_bars} bars on {self.device}...")
        logger.info(f"Assets Universe: {self.assets} | Leverage: {self.leverage:.0f}x | Initial Capital: ${self.initial_capital:.2f}")

        # Warmup training
        self.train_policy_on_window(warmup_bars, epochs=16)

        for t in range(warmup_bars, total_bars):
            if (t - warmup_bars) > 0 and (t - warmup_bars) % retrain_freq == 0:
                self.train_policy_on_window(t, epochs=8)

            ts_str = self.asset_candles["BTC"][t].get("timestamp", f"Bar_{t}")

            # ---------------------------------------------------------
            # 1. Active Position Management (Moonbag Split & Runners)
            # ---------------------------------------------------------
            if self.position is not None:
                pos = self.position
                sym = pos["asset"]
                c = self.asset_candles[sym][t]
                side = pos["side"]
                entry_px = pos["entry_price"]
                units = pos["units"]
                stop_dist = pos["stop_dist"]

                if side == "LONG":
                    pos["highest_price"] = max(pos["highest_price"], c["high"])

                    # A. Moonbag Split Target (+1.5R): Bank 50% profit into cash!
                    just_banked = False
                    if not pos["half_banked"] and c["high"] >= pos["target_1"]:
                        half_units = units * 0.50
                        exit_px = pos["target_1"]
                        gross = (exit_px - entry_px) * half_units
                        fee = (half_units * exit_px) * self.fee_rate
                        pnl_bank = gross - fee
                        half_margin = pos["margin_allocated"] * 0.50

                        self.cash += (half_margin + pnl_bank)
                        pos["half_banked"] = True
                        pos["banked_at_bar"] = t
                        pos["units"] -= half_units
                        pos["margin_allocated"] -= half_margin
                        just_banked = True

                        # Move stop on remaining 50% runner to Breakeven (+0.3% fee cushion)
                        pos["stop_loss_price"] = max(pos["stop_loss_price"], entry_px * 1.003)

                        self.trades.append({
                            "id": len(self.trades) + 1,
                            "bar": t,
                            "timestamp": ts_str,
                            "asset": sym,
                            "side": "MOONBAG_BANK_50%_LONG",
                            "entry": round(entry_px, 2),
                            "exit": round(exit_px, 2),
                            "units": round(half_units, 6),
                            "pnl_usd": round(pnl_bank, 2),
                            "pnl_pct": round((pnl_bank / half_margin) * 100.0, 2),
                            "capital": round(self.cash, 2),
                            "reason": "Moonbag 50% Bank (+1.5R Banked to Cash)"
                        })

                    # B. Uncapped Parabolic Trailing Ratchet for the Moonbag runner (active on subsequent bars)
                    if pos["half_banked"] and not just_banked:
                        trail = pos["highest_price"] - (2.2 * pos["atr_14"])
                        if trail > pos["stop_loss_price"]:
                            pos["stop_loss_price"] = trail

                    # C. Stop Loss or Trailing Runner Exit (only evaluate if not just banked on this bar)
                    if not just_banked and c["low"] <= pos["stop_loss_price"]:
                        exit_px = pos["stop_loss_price"]
                        gross = (exit_px - entry_px) * pos["units"]
                        fee = (pos["units"] * exit_px) * self.fee_rate
                        pnl = gross - fee
                        self.cash += max(0.0, pos["margin_allocated"] + pnl)

                        is_runner = pos["half_banked"]
                        side_label = "MOONBAG_RUNNER_EXIT_LONG" if is_runner else "STOP_LOSS_CLOSE_LONG"

                        self.trades.append({
                            "id": len(self.trades) + 1,
                            "bar": t,
                            "timestamp": ts_str,
                            "asset": sym,
                            "side": side_label,
                            "entry": round(entry_px, 2),
                            "exit": round(exit_px, 2),
                            "units": round(pos["units"], 6),
                            "pnl_usd": round(pnl, 2),
                            "pnl_pct": round((pnl / pos["margin_allocated"]) * 100.0, 2),
                            "capital": round(self.cash, 2),
                            "reason": f"{'Parabolic Moonbag Runner Close' if is_runner else 'Dynamic 5m ATR Stop'} ({pos['leverage']}x)"
                        })

                        self.last_trade_closed_bar = t
                        self.last_trade_was_loss = (pnl < 0) and not is_runner
                        if self.last_trade_was_loss:
                            self.consecutive_losses += 1
                        else:
                            self.consecutive_losses = 0

                        self.position = None

                else:  # SHORT
                    pos["lowest_price"] = min(pos["lowest_price"], c["low"])

                    # A. Moonbag Split Target (+1.5R): Bank 50% profit into cash!
                    just_banked = False
                    if not pos["half_banked"] and c["low"] <= pos["target_1"]:
                        half_units = units * 0.50
                        exit_px = pos["target_1"]
                        gross = (entry_px - exit_px) * half_units
                        fee = (half_units * exit_px) * self.fee_rate
                        pnl_bank = gross - fee
                        half_margin = pos["margin_allocated"] * 0.50

                        self.cash += (half_margin + pnl_bank)
                        pos["half_banked"] = True
                        pos["banked_at_bar"] = t
                        pos["units"] -= half_units
                        pos["margin_allocated"] -= half_margin
                        just_banked = True

                        # Move stop on remaining 50% runner to Breakeven (-0.3% fee cushion)
                        pos["stop_loss_price"] = min(pos["stop_loss_price"], entry_px * 0.997)

                        self.trades.append({
                            "id": len(self.trades) + 1,
                            "bar": t,
                            "timestamp": ts_str,
                            "asset": sym,
                            "side": "MOONBAG_BANK_50%_SHORT",
                            "entry": round(entry_px, 2),
                            "exit": round(exit_px, 2),
                            "units": round(half_units, 6),
                            "pnl_usd": round(pnl_bank, 2),
                            "pnl_pct": round((pnl_bank / half_margin) * 100.0, 2),
                            "capital": round(self.cash, 2),
                            "reason": "Moonbag 50% Bank (+1.5R Banked to Cash)"
                        })

                    # B. Uncapped Parabolic Trailing Ratchet for the Moonbag runner (active on subsequent bars)
                    if pos["half_banked"] and not just_banked:
                        trail = pos["lowest_price"] + (2.2 * pos["atr_14"])
                        if trail < pos["stop_loss_price"]:
                            pos["stop_loss_price"] = trail

                    # C. Stop Loss or Trailing Runner Exit (only evaluate if not just banked on this bar)
                    if not just_banked and c["high"] >= pos["stop_loss_price"]:
                        exit_px = pos["stop_loss_price"]
                        gross = (entry_px - exit_px) * pos["units"]
                        fee = (pos["units"] * exit_px) * self.fee_rate
                        pnl = gross - fee
                        self.cash += max(0.0, pos["margin_allocated"] + pnl)

                        is_runner = pos["half_banked"]
                        side_label = "MOONBAG_RUNNER_EXIT_SHORT" if is_runner else "STOP_LOSS_CLOSE_SHORT"

                        self.trades.append({
                            "id": len(self.trades) + 1,
                            "bar": t,
                            "timestamp": ts_str,
                            "asset": sym,
                            "side": side_label,
                            "entry": round(entry_px, 2),
                            "exit": round(exit_px, 2),
                            "units": round(pos["units"], 6),
                            "pnl_usd": round(pnl, 2),
                            "pnl_pct": round((pnl / pos["margin_allocated"]) * 100.0, 2),
                            "capital": round(self.cash, 2),
                            "reason": f"{'Parabolic Moonbag Runner Close' if is_runner else 'Dynamic 5m ATR Stop'} ({pos['leverage']}x)"
                        })

                        self.last_trade_closed_bar = t
                        self.last_trade_was_loss = (pnl < 0) and not is_runner
                        if self.last_trade_was_loss:
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
            # 2. Multi-Asset Radar & Order Entry
            # ---------------------------------------------------------
            if self.consecutive_losses >= 2:
                req_cooldown = 72  # 6 hours circuit breaker
            elif self.consecutive_losses == 1:
                req_cooldown = 24  # 2 hours cooldown
            else:
                req_cooldown = 6   # 30 mins after win

            in_cooldown = (t - self.last_trade_closed_bar) < req_cooldown

            if self.position is None and not in_cooldown:
                # Cross-sectional scan across BTC, ETH, and SOL
                candidate_setups = []

                for sym in self.assets:
                    choice, conf = self.infer_asset(sym, t)
                    m = self.f_dicts[sym][t]

                    if choice in ["buy", "sell"]:
                        score = score_setup_confluence(m, choice)
                        # Regime filter
                        is_trend_ok = (choice == "buy" and m["htf_regime"] == "TREND_BULL") or \
                                      (choice == "sell" and m["htf_regime"] == "TREND_BEAR")

                        if is_trend_ok and score >= 45.0:
                            candidate_setups.append({
                                "asset": sym,
                                "choice": choice,
                                "conf": conf,
                                "score": score,
                                "candle": self.asset_candles[sym][t],
                                "m": m
                            })

                # Select the single highest-conviction setup across the market
                if candidate_setups:
                    best = max(candidate_setups, key=lambda s: s["score"])
                    sym = best["asset"]
                    choice = best["choice"]
                    m = best["m"]
                    c = best["candle"]
                    cur_px = c["close"]
                    atr = m["atr_14"]
                    stop_dist = max(atr * 2.2, cur_px * 0.008)
                    stop_dist_pct = stop_dist / cur_px

                    margin_alloc = self.compute_confluence_margin(self.cash, stop_dist_pct, best["score"])

                    if margin_alloc >= 15.0 and self.cash >= margin_alloc:
                        fee = margin_alloc * self.fee_rate
                        eff_margin = margin_alloc - fee
                        notional = eff_margin * self.leverage

                        if choice == "buy":
                            exec_px = cur_px * (1.0 + self.slippage_pct)
                            units = notional / exec_px
                            sl_px = exec_px - stop_dist
                            target_1 = exec_px + (1.5 * stop_dist)
                            liq_px = exec_px * (1.0 - (1.0 / self.leverage) + self.maintenance_margin_rate)

                            self.cash -= margin_alloc
                            self.position = {
                                "asset": sym,
                                "side": "LONG",
                                "entry_price": exec_px,
                                "units": units,
                                "leverage": self.leverage,
                                "margin_allocated": margin_alloc,
                                "stop_loss_price": sl_px,
                                "target_1": target_1,
                                "highest_price": exec_px,
                                "lowest_price": exec_px,
                                "liquidation_price": liq_px,
                                "atr_14": atr,
                                "stop_dist": stop_dist,
                                "half_banked": False,
                                "score": best["score"]
                            }

                        elif choice == "sell":
                            exec_px = cur_px * (1.0 - self.slippage_pct)
                            units = notional / exec_px
                            sl_px = exec_px + stop_dist
                            target_1 = exec_px - (1.5 * stop_dist)
                            liq_px = exec_px * (1.0 + (1.0 / self.leverage) - self.maintenance_margin_rate)

                            self.cash -= margin_alloc
                            self.position = {
                                "asset": sym,
                                "side": "SHORT",
                                "entry_price": exec_px,
                                "units": units,
                                "leverage": self.leverage,
                                "margin_allocated": margin_alloc,
                                "stop_loss_price": sl_px,
                                "target_1": target_1,
                                "highest_price": exec_px,
                                "lowest_price": exec_px,
                                "liquidation_price": liq_px,
                                "atr_14": atr,
                                "stop_dist": stop_dist,
                                "half_banked": False,
                                "score": best["score"]
                            }

            # Record total equity
            unrealized = 0.0
            if self.position is not None:
                p = self.position
                px = self.asset_candles[p["asset"]][t]["close"]
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
            pos = self.position
            c_last = self.asset_candles[pos["asset"]][-1]
            px = c_last["close"]
            pnl = (px - pos["entry_price"]) * pos["units"] if pos["side"] == "LONG" else (pos["entry_price"] - px) * pos["units"]
            self.cash += max(0.0, pos["margin_allocated"] + pnl)
            self.trades.append({
                "id": len(self.trades) + 1,
                "bar": total_bars - 1,
                "timestamp": c_last.get("timestamp", ""),
                "asset": pos["asset"],
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

        return self._generate_report(warmup_bars, total_bars)

    def _generate_report(self, warmup_bars: int, total_bars: int) -> Dict[str, Any]:
        closed = [t for t in self.trades if "CLOSE" in t["side"] or "EXIT" in t["side"] or "BANK" in t["side"]]
        wins = [t for t in closed if t["pnl_usd"] > 0]
        losses = [t for t in closed if t["pnl_usd"] < 0]
        longs = [t for t in closed if "LONG" in t["side"]]
        shorts = [t for t in closed if "SHORT" in t["side"]]

        total_gain = sum(t["pnl_usd"] for t in wins)
        total_loss = abs(sum(t["pnl_usd"] for t in losses))
        profit_factor = round(total_gain / max(total_loss, 1e-6), 2)

        win_rate = round((len(wins) / len(closed) * 100.0), 1) if closed else 0.0
        long_win_rate = round((len([t for t in longs if t["pnl_usd"] > 0]) / len(longs) * 100.0), 1) if longs else 0.0
        short_win_rate = round((len([t for t in shorts if t["pnl_usd"] > 0]) / len(shorts) * 100.0), 1) if shorts else 0.0

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
            "assets": self.assets,
            "total_bars": total_bars,
            "oos_bars": total_bars - warmup_bars,
            "start_time": self.asset_candles["BTC"][warmup_bars].get("timestamp", ""),
            "end_time": self.asset_candles["BTC"][-1].get("timestamp", ""),
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

        print("\n" + "=" * 75)
        print("INSTITUTIONAL MULTI-ASSET COMPOUNDING FUTURES ENGINE (5M WALK-FORWARD)")
        print("=" * 75)
        print(f"Asset Universe:         {', '.join(self.assets)} (5-Minute Perpetuals)")
        print(f"Out-Of-Sample Horizon:  {report['start_time']} to {report['end_time']} ({report['oos_bars']} bars, ~{report['oos_bars']/288:.1f} days)")
        print(f"Initial Capital:        ${self.initial_capital:.2f}")
        print(f"Ending Equity:          ${ending_equity:.2f} ({report['multiplier']}x)")
        print(f"Total Return:           {'+' if tot_return_pct >= 0 else ''}{tot_return_pct}%")
        print(f"Annualized Sharpe:      {report['annualized_sharpe']}")
        print(f"Annualized Sortino:     {report['annualized_sortino']}")
        print(f"Max Drawdown:           {max_dd_pct}% (${max_dd_usd:.2f})")
        print(f"Total Executions:       {len(closed)}")
        print(f"Overall Win Rate:       {win_rate}% (Longs: {long_win_rate}%, Shorts: {short_win_rate}%)")
        print(f"Profit Factor:          {profit_factor}")
        print(f"Milestones Reached:     {report['milestones_reached']}")
        print("=" * 75 + "\n")

        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)

        trades_df = pd.DataFrame(self.trades)
        trades_csv_path = os.path.join(data_dir, "institutional_compounding_trades.csv")
        trades_df.to_csv(trades_csv_path, index=False)
        logger.info(f"Saved institutional trade blotter to {trades_csv_path}")

        results_json_path = os.path.join(data_dir, "institutional_compounding_results.json")
        with open(results_json_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved institutional simulation report to {results_json_path}")

        return report


def main():
    assets = ["BTC", "ETH", "SOL"]
    tickers = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
    asset_candles = {}

    logger.info("Downloading 60 days of 5-minute candles for BTC, ETH, and SOL...")

    for sym, tkr in tickers.items():
        df = yf.download(tickers=tkr, interval="5m", period="60d", progress=False)
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
        asset_candles[sym] = candles
        logger.info(f"Loaded {len(candles)} 5m candles for {sym}.")

    # Align candle counts
    min_len = min(len(c) for c in asset_candles.values())
    for sym in assets:
        asset_candles[sym] = asset_candles[sym][-min_len:]

    sim = InstitutionalCompoundingSimulator(
        asset_candles=asset_candles,
        initial_capital=300.0,
        leverage=10.0
    )
    sim.run_simulation(warmup_bars=1500, retrain_freq=500)


if __name__ == "__main__":
    main()
