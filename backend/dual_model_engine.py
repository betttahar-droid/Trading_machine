"""
LayaQuant Dual-Model Collaborative Intelligence Engine
Hierarchical Architecture combining:
- Model A: LayaMacroGovernor (System 2 - Slow Brain: 4H/Daily Regimes, News Sentiment, Directional Mandates, Leverage Cap, Kelly Risk Budget)
- Model B: LayaMicroSniper (System 1 - Fast Brain: 5m/1m Bollinger-Keltner Squeezes, CVD Order Flow, Precision Execution, Trailing Ratchet)
- DualConsensusSynthesizer: Consensus verification with Governor Veto Armor.
"""

import time
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("layaquant.dual_model")


class LayaMacroGovernor:
    """
    Model A: Macro Strategist & Risk Governor (System 2 - Slow Brain)
    Analyzes higher-timeframe market structure, volatility clustering, macro regime,
    and real-time news sentiment.
    Outputs:
    - directional_permit: "PERMIT_LONG", "PERMIT_SHORT", "PERMIT_BOTH", or "ENFORCE_CASH"
    - max_leverage_allowed: Dynamic leverage ceiling (1x to 10x)
    - macro_risk_budget_pct: Dynamic Kelly fraction cap (1.0% to 2.5%)
    - macro_regime: "TREND_BULL", "TREND_BEAR", "RANGE_BOUND", "TOXIC_CHOP"
    - macro_confidence: 0.0 to 1.0
    - veto_active: bool
    """
    def __init__(self):
        self.name = "Laya-Macro-Governor-v2"
        self.role = "Macro Strategist & Risk Governor (System 2)"
        self.last_evaluation = {}

    def evaluate(self, 
                 htf_regime: str, 
                 news_score: float = 0.0, 
                 atr_expansion: float = 1.0, 
                 current_price: float = 0.0, 
                 ticker_sentiment: Optional[Dict[str, Any]] = None,
                 fear_and_greed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate higher-timeframe safety, news catalysts, and issue directional permits.
        Dynamically adapts leverage ceilings and target multipliers based on Andrew Lo's
        Adaptive Markets Hypothesis and Extreme Fear & Greed regimes.
        """
        # Asset-specific news catalyst override
        top_catalyst = ticker_sentiment.get("top_catalyst", "") if ticker_sentiment else ""
        asset_score = ticker_sentiment.get("score", news_score) if ticker_sentiment else news_score
        is_breaking = ticker_sentiment.get("breaking", False) if ticker_sentiment else False

        # Adaptive Sentiment Metrics (Alternative.me Index)
        fng_val = int(fear_and_greed.get("value", 50)) if fear_and_greed else 50
        fng_label = fear_and_greed.get("value_classification", "Neutral") if fear_and_greed else "Neutral"
        is_extreme_sentiment = (fng_val >= 75 or fng_val <= 25)
        tp_multiplier = 3.2 if is_extreme_sentiment else 2.5
        sl_multiplier = 2.4 if is_extreme_sentiment else 2.2

        # 1. Hard Circuit Breaker for Toxic Liquidation Chop
        if htf_regime == "TOXIC_CHOP":
            result = {
                "model": "Model A (Macro Governor)",
                "macro_regime": "TOXIC_CHOP",
                "directional_permit": "ENFORCE_CASH",
                "veto_active": True,
                "veto_reason": "High-volatility toxic chop detected on 4H macro frame. Enforcing 100% Cash defense.",
                "max_leverage_allowed": 1.0,
                "macro_risk_budget_pct": 0.0,
                "macro_confidence": 0.95,
                "sentiment_alignment": "NEUTRAL",
                "top_catalyst": top_catalyst,
                "timestamp": time.time()
            }
            self.last_evaluation = result
            return result

        # 2. Asset-Specific Breaking Negative News Veto (FUD / Exploit / SEC Veto)
        if asset_score < -0.45:
            result = {
                "model": "Model A (Macro Governor)",
                "macro_regime": htf_regime,
                "directional_permit": "ENFORCE_CASH" if htf_regime == "TREND_BULL" else "PERMIT_SHORT",
                "veto_active": True,
                "veto_reason": f"Asset FUD Catalyst ({top_catalyst[:80]}). Long trades blocked to protect capital.",
                "max_leverage_allowed": 3.0,
                "macro_risk_budget_pct": 0.01,
                "macro_confidence": 0.85,
                "sentiment_alignment": "CRITICAL_BEARISH",
                "top_catalyst": top_catalyst,
                "summary": f"Governor Veto on Longs: Bearish news headwind ({asset_score:+.2f}).",
                "timestamp": time.time()
            }
            self.last_evaluation = result
            return result

        # 3. Bullish Macro Regime
        if htf_regime == "TREND_BULL":
            if asset_score < -0.20:
                lev = 3.0
                permit = "PERMIT_BOTH"
                veto = False
                reason = "4H Bull trend intact, but mild negative catalyst warrants conservative 3x leverage."
                conf = 0.68
            else:
                lev = 10.0 if (atr_expansion < 1.4 or asset_score > 0.4) else 5.0
                permit = "PERMIT_LONG"
                veto = False
                boost_msg = f" (Catalyst Boost: {top_catalyst[:60]})" if asset_score > 0.4 else ""
                reason = f"Clean 4H Bullish Structure (50/200 EMA Golden Alignment){boost_msg}. Long trades prioritized."
                conf = 0.92 if asset_score > 0.4 else 0.88

            # Andrew Lo AMH Extreme Sentiment Throttle
            if is_extreme_sentiment and lev > 6.0:
                lev = 6.0
                reason += f" [AMH Volatility Guard: {fng_val} {fng_label} throttles leverage to 6x max]"

            result = {
                "model": "Model A (Macro Governor)",
                "macro_regime": "TREND_BULL",
                "directional_permit": permit,
                "veto_active": veto,
                "veto_reason": None,
                "max_leverage_allowed": lev,
                "macro_risk_budget_pct": 0.025 if lev >= 10.0 else 0.015,
                "macro_confidence": conf,
                "sentiment_alignment": "POSITIVE" if asset_score > 0 else "NEUTRAL",
                "top_catalyst": top_catalyst,
                "fear_and_greed_val": fng_val,
                "fear_and_greed_label": fng_label,
                "is_extreme_sentiment": is_extreme_sentiment,
                "tp_multiplier": tp_multiplier,
                "sl_multiplier": sl_multiplier,
                "summary": reason,
                "timestamp": time.time()
            }
            self.last_evaluation = result
            return result

        # 4. Bearish Macro Regime
        if htf_regime == "TREND_BEAR":
            if asset_score > 0.35:
                lev = 3.0
                permit = "PERMIT_BOTH"
                veto = False
                reason = "4H Bear trend, but positive news/whale catalyst warrants conservative 3x leverage."
                conf = 0.65
            else:
                lev = 10.0 if atr_expansion < 1.4 else 5.0
                permit = "PERMIT_SHORT"
                veto = False
                reason = "Clean 4H Bearish Structure (Death Cross Alignment). Short trades prioritized."
                conf = 0.85

            if is_extreme_sentiment and lev > 6.0:
                lev = 6.0
                reason += f" [AMH Volatility Guard: {fng_val} {fng_label} throttles leverage to 6x max]"

            result = {
                "model": "Model A (Macro Governor)",
                "macro_regime": "TREND_BEAR",
                "directional_permit": permit,
                "veto_active": veto,
                "veto_reason": None,
                "max_leverage_allowed": lev,
                "macro_risk_budget_pct": 0.025 if lev >= 10.0 else 0.015,
                "macro_confidence": conf,
                "sentiment_alignment": "NEGATIVE" if news_score < 0 else "NEUTRAL",
                "fear_and_greed_val": fng_val,
                "fear_and_greed_label": fng_label,
                "is_extreme_sentiment": is_extreme_sentiment,
                "tp_multiplier": tp_multiplier,
                "sl_multiplier": sl_multiplier,
                "summary": reason,
                "timestamp": time.time()
            }
            self.last_evaluation = result
            return result

        # 5. Range-Bound / Neutral Macro Regime
        result = {
            "model": "Model A (Macro Governor)",
            "macro_regime": "RANGE_BOUND",
            "directional_permit": "PERMIT_BOTH",
            "veto_active": False,
            "veto_reason": None,
            "max_leverage_allowed": 5.0,  # Cap at 5x in ranges
            "macro_risk_budget_pct": 0.015,
            "macro_confidence": 0.60,
            "sentiment_alignment": "NEUTRAL",
            "fear_and_greed_val": fng_val,
            "fear_and_greed_label": fng_label,
            "is_extreme_sentiment": is_extreme_sentiment,
            "tp_multiplier": tp_multiplier,
            "sl_multiplier": sl_multiplier,
            "summary": "4H Range-bound consolidation. Mean-reversion scalping permitted at 5x leverage max.",
            "timestamp": time.time()
        }
        self.last_evaluation = result
        return result


class LayaMicroSniper:
    """
    Model B: Tactical Micro Sniper & Executioner (System 1 - Fast Brain)
    Analyzes 5-minute / 1-minute microstructure, Bollinger-Keltner Volatility Squeezes,
    CVD delta imbalances, and ATR wick dynamics.
    Outputs:
    - tactical_action: "BUY", "SELL", or "HOLD"
    - tactical_confidence: 0.0 to 1.0
    - squeeze_state: "SQUEEZE_ON", "SQUEEZE_FIRE", "NORMAL"
    - cvd_momentum: float
    - proposed_stop_dist: float
    - proposed_target_dist: float
    - summary: str
    """
    def __init__(self):
        self.name = "Laya-Micro-Sniper-v1"
        self.role = "Tactical Micro Sniper & Executioner (System 1)"
        self.last_evaluation = {}

    def evaluate(self, 
                 candles_5m: List[Dict[str, Any]], 
                 atr_14: float, 
                 cvd_delta: float, 
                 rsi: float, 
                 spread_pct: float,
                 depth_ratio: float = 1.0,
                 spread_bps: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate 5-minute intraday setup, detect volatility squeeze breakouts,
        and enforce Level-2 Order Flow Imbalance (OFI) and VPIN Flow Toxicity guards.
        """
        if not candles_5m or len(candles_5m) < 15:
            return {
                "model": "Model B (Micro Sniper)",
                "tactical_action": "HOLD",
                "tactical_confidence": 0.50,
                "squeeze_state": "NORMAL",
                "vpin_toxicity": "CLEAN",
                "depth_ratio": depth_ratio,
                "summary": "Insufficient candle depth for micro evaluation."
            }

        cur_px = float(candles_5m[-1]["close"])
        closes = [float(c["close"]) for c in candles_5m]
        
        # Bollinger Bands (20, 2.0 std)
        n = min(20, len(closes))
        sma20 = float(np.mean(closes[-n:]))
        std20 = float(np.std(closes[-n:])) if n > 1 else (cur_px * 0.01)
        bb_upper = sma20 + (2.0 * std20)
        bb_lower = sma20 - (2.0 * std20)

        # Keltner Channels (20, 1.5 ATR)
        kc_upper = sma20 + (1.5 * atr_14)
        kc_lower = sma20 - (1.5 * atr_14)

        # Volatility Squeeze condition (BB inside KC)
        is_squeeze = bool(bb_upper < kc_upper and bb_lower > kc_lower)
        squeeze_state = "SQUEEZE_ON" if is_squeeze else "NORMAL"

        # Directional Micro Momentum
        action = "HOLD"
        confidence = 0.50
        reason_parts = []

        # Bullish Micro Signal
        if spread_pct > 0.06 and cvd_delta > 0:
            action = "BUY"
            confidence = 0.72 + (0.15 if is_squeeze else 0.0) + (0.08 if 45 <= rsi <= 65 else 0.0)
            confidence = min(0.95, confidence)
            reason_parts.append(f"Fast SMA > Slow SMA (+{spread_pct:.2f}%)")
            reason_parts.append(f"CVD Aggressive Buying (+{cvd_delta:.1f})")
            if is_squeeze:
                reason_parts.append("⚡ 5m Volatility Squeeze Active (Coiling for Breakout)")
        
        # Bearish Micro Signal
        elif spread_pct < -0.06 and cvd_delta < 0:
            action = "SELL"
            confidence = 0.72 + (0.15 if is_squeeze else 0.0) + (0.08 if 35 <= rsi <= 55 else 0.0)
            confidence = min(0.95, confidence)
            reason_parts.append(f"Fast SMA < Slow SMA ({spread_pct:.2f}%)")
            reason_parts.append(f"CVD Aggressive Selling ({cvd_delta:.1f})")
            if is_squeeze:
                reason_parts.append("⚡ 5m Volatility Squeeze Active (Coiling for Breakdown)")
        
        else:
            action = "HOLD"
            confidence = 0.52
            reason_parts.append(f"Neutral micro-order flow (Spread: {spread_pct:+.2f}%, CVD: {cvd_delta:+.1f})")

        # -------------------------------------------------------------
        # 1. VPIN Flow Toxicity & Absorption Guard (Easley, Lopez de Prado & O'Hara 2012)
        # -------------------------------------------------------------
        vpin_toxicity = "CLEAN"
        vpin_veto = False
        last_c = candles_5m[-1]
        c_open = float(last_c.get("open", cur_px))
        c_high = float(last_c.get("high", cur_px))
        c_low = float(last_c.get("low", cur_px))
        c_close = float(last_c.get("close", cur_px))
        c_vol = float(last_c.get("volume", 0.0))
        c_range = max(c_high - c_low, cur_px * 0.0005)

        vols = [float(c.get("volume", 0.0)) for c in candles_5m]
        avg_vol = float(np.mean(vols[-20:])) if len(vols) >= 20 else max(1.0, c_vol)
        rel_vol = c_vol / max(avg_vol, 1e-4)

        upper_wick = c_high - max(c_open, c_close)
        lower_wick = min(c_open, c_close) - c_low
        upper_wick_ratio = upper_wick / c_range
        lower_wick_ratio = lower_wick / c_range

        # Bearish Absorption: High volume upper wick rejection (whales absorbing aggressive retail buys)
        if rel_vol >= 2.0 and upper_wick_ratio >= 0.42:
            vpin_toxicity = "BEARISH_ABSORPTION"
            if action == "BUY":
                action = "HOLD"
                vpin_veto = True
                confidence = 0.40
                reason_parts.append(f"⚠️ VPIN Toxicity: Bearish Absorption detected ({upper_wick_ratio*100:.0f}% upper wick on {rel_vol:.1f}x volume)")

        # Bullish Absorption: High volume lower wick rejection (whales absorbing aggressive retail sells)
        elif rel_vol >= 2.0 and lower_wick_ratio >= 0.42:
            vpin_toxicity = "BULLISH_ABSORPTION"
            if action == "SELL":
                action = "HOLD"
                vpin_veto = True
                confidence = 0.40
                reason_parts.append(f"⚠️ VPIN Toxicity: Bullish Absorption detected ({lower_wick_ratio*100:.0f}% lower wick on {rel_vol:.1f}x volume)")

        # -------------------------------------------------------------
        # 2. Order Flow Imbalance (OFI) & L2 Book Depth Filter (Cont & Stoikov 2014)
        # -------------------------------------------------------------
        ofi_veto = False
        if action == "BUY":
            if depth_ratio < 0.85:
                action = "HOLD"
                ofi_veto = True
                confidence = 0.45
                reason_parts.append(f"🚫 OFI Veto: Hollow bid book (Depth ratio {depth_ratio:.2f} < 0.85). Asks outweigh bids by {((1.0/max(0.1, depth_ratio))-1.0)*100:.0f}%.")
            elif depth_ratio >= 1.20:
                confidence = min(0.98, confidence + 0.05)
                reason_parts.append(f"📊 Heavy Bid Depth Support ({depth_ratio:.2f}x Bids)")
        elif action == "SELL":
            if depth_ratio > 1.15:
                action = "HOLD"
                ofi_veto = True
                confidence = 0.45
                reason_parts.append(f"🚫 OFI Veto: Heavy bid wall (Depth ratio {depth_ratio:.2f} > 1.15). Dangerous to short into resting support.")
            elif depth_ratio <= 0.80:
                confidence = min(0.98, confidence + 0.05)
                reason_parts.append(f"📊 Thin Bid Depth ({depth_ratio:.2f}x Asks dominant)")

        stop_dist = max(2.2 * atr_14, cur_px * 0.008)
        target_dist = 2.5 * stop_dist

        result = {
            "model": "Model B (Micro Sniper)",
            "tactical_action": action,
            "tactical_confidence": round(confidence, 3),
            "squeeze_state": squeeze_state,
            "is_squeeze": is_squeeze,
            "cvd_momentum": round(cvd_delta, 1),
            "rsi": round(rsi, 1),
            "depth_ratio": round(depth_ratio, 2),
            "spread_bps": round(spread_bps, 2),
            "vpin_toxicity": vpin_toxicity,
            "vpin_veto": vpin_veto,
            "ofi_veto": ofi_veto,
            "proposed_stop_dist": round(stop_dist, 4 if cur_px < 1.0 else 2),
            "proposed_target_dist": round(target_dist, 4 if cur_px < 1.0 else 2),
            "summary": " | ".join(reason_parts),
            "timestamp": time.time()
        }
        self.last_evaluation = result
        return result


class DualConsensusSynthesizer:
    """
    Dual-Model Consensus Synthesizer
    Combines Model A (Macro Governor) and Model B (Micro Sniper) into an authoritative,
    high-conviction trading decision.
    Enforces Governor Veto Armor to eliminate false breakouts and whipsaws.
    """
    def __init__(self):
        self.governor = LayaMacroGovernor()
        self.sniper = LayaMicroSniper()
        self.enforce_dual_consensus = True
        self.min_consensus_confidence = 0.68
        self.recent_syntheses = []

    def evaluate_opportunity(self, 
                             symbol: str, 
                             htf_regime: str, 
                             candles_5m: List[Dict[str, Any]], 
                             atr_14: float, 
                             cvd_delta: float, 
                             rsi: float, 
                             spread_pct: float, 
                             news_score: float = 0.0,
                             atr_expansion: float = 1.0,
                             ticker_sentiment: Optional[Dict[str, Any]] = None,
                             depth_ratio: float = 1.0,
                             spread_bps: float = 0.0,
                             fear_and_greed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synthesize Model A and Model B for a specific ticker with asset-level sentiment,
        L2 order book depth imbalance, VPIN toxicity guards, and adaptive market regimes.
        """
        # Step 1: Model A Macro Evaluation (with Ticker-Specific News & Adaptive Sentiment)
        macro_out = self.governor.evaluate(
            htf_regime=htf_regime,
            news_score=news_score,
            atr_expansion=atr_expansion,
            ticker_sentiment=ticker_sentiment,
            fear_and_greed=fear_and_greed
        )

        # Step 2: Model B Tactical Evaluation (with L2 OFI Depth & VPIN Flow Toxicity)
        micro_out = self.sniper.evaluate(
            candles_5m=candles_5m,
            atr_14=atr_14,
            cvd_delta=cvd_delta,
            rsi=rsi,
            spread_pct=spread_pct,
            depth_ratio=depth_ratio,
            spread_bps=spread_bps
        )

        # Step 3: Consensus Protocol & Veto Verification
        tactical_action = micro_out["tactical_action"]
        permit = macro_out["directional_permit"]
        macro_veto = macro_out["veto_active"]
        
        consensus_status = "NO_SIGNAL"
        final_action = "HOLD"
        consensus_score = 0.0
        synthesis_reasoning = ""

        # Check for Governor Hard Veto
        if macro_veto:
            consensus_status = "MACRO_GOVERNOR_VETO"
            final_action = "HOLD"
            synthesis_reasoning = f"VETO: {macro_out.get('veto_reason', 'Macro regime unsafe')}"
            consensus_score = 15.0

        elif tactical_action == "HOLD":
            consensus_status = "MICRO_HUNTING"
            final_action = "HOLD"
            synthesis_reasoning = f"Model B is hunting for a 5m squeeze/breakout setup. Model A permits {permit}."
            consensus_score = 30.0

        # Check Buy Consensus
        elif tactical_action == "BUY":
            if permit in ["PERMIT_LONG", "PERMIT_BOTH"]:
                consensus_status = "DUAL_CONSENSUS_ACHIEVED"
                final_action = "BUY_LONG"
                consensus_score = 50.0 + (macro_out["macro_confidence"] * 25.0) + (micro_out["tactical_confidence"] * 25.0)
                synthesis_reasoning = (
                    f"DUAL AGREEMENT: Model A confirms {macro_out['macro_regime']} ({macro_out['macro_confidence']*100:.0f}%), "
                    f"Model B confirms 5m Breakout ({micro_out['tactical_confidence']*100:.0f}%). "
                    f"Max Leverage: {macro_out['max_leverage_allowed']:.0f}x."
                )
            else:
                consensus_status = "DIRECTIONAL_MISMATCH_VETO"
                final_action = "HOLD"
                consensus_score = 25.0
                synthesis_reasoning = f"VETO: Model B wants BUY, but Model A only permits {permit} (Counter-trend danger prevented)."

        # Check Sell Consensus
        elif tactical_action == "SELL":
            if permit in ["PERMIT_SHORT", "PERMIT_BOTH"]:
                consensus_status = "DUAL_CONSENSUS_ACHIEVED"
                final_action = "SELL_SHORT"
                consensus_score = 50.0 + (macro_out["macro_confidence"] * 25.0) + (micro_out["tactical_confidence"] * 25.0)
                synthesis_reasoning = (
                    f"DUAL AGREEMENT: Model A confirms {macro_out['macro_regime']} ({macro_out['macro_confidence']*100:.0f}%), "
                    f"Model B confirms 5m Breakdown ({micro_out['tactical_confidence']*100:.0f}%). "
                    f"Max Leverage: {macro_out['max_leverage_allowed']:.0f}x."
                )
            else:
                consensus_status = "DIRECTIONAL_MISMATCH_VETO"
                final_action = "HOLD"
                consensus_score = 25.0
                synthesis_reasoning = f"VETO: Model B wants SELL, but Model A only permits {permit} (Counter-trend danger prevented)."

        dual_telemetry = {
            "symbol": symbol,
            "final_action": final_action,
            "consensus_status": consensus_status,
            "consensus_score": round(consensus_score, 1),
            "dual_consensus_achieved": (consensus_status == "DUAL_CONSENSUS_ACHIEVED"),
            "model_a_macro": macro_out,
            "model_b_micro": micro_out,
            "ticker_sentiment": ticker_sentiment or {},
            "depth_ratio": micro_out.get("depth_ratio", 1.0),
            "vpin_toxicity": micro_out.get("vpin_toxicity", "CLEAN"),
            "recommended_leverage": macro_out["max_leverage_allowed"],
            "recommended_risk_pct": macro_out["macro_risk_budget_pct"],
            "tp_multiplier": macro_out.get("tp_multiplier", 2.5),
            "sl_multiplier": macro_out.get("sl_multiplier", 2.2),
            "is_extreme_sentiment": macro_out.get("is_extreme_sentiment", False),
            "synthesis_reasoning": synthesis_reasoning,
            "timestamp": time.time()
        }

        return dual_telemetry


# Global Singleton Instance
dual_model_engine = DualConsensusSynthesizer()
