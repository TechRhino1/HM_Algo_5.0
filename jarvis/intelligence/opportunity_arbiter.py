"""
HM Algo 2.0 — Universal Master Opportunity Arbiter & Autonomous Trade Selector.
Evaluates, scores, and ranks trading opportunities across multi-asset universes and multiple trading styles
(SWING, DAY_TRADING, SCALP) using Machine Learning probability, Expected Value (EV), Master Confluence,
Adversarial Threat Penalty, and Market Regime dynamic multipliers.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np

from jarvis.data.schemas import DecisionObject, MarketContext
from jarvis.intelligence.gate_registry import HARD_GATES, canonical_gate

logger = logging.getLogger("JARVIS_OpportunityArbiter")

@dataclass
class CandidateOpportunity:
    """
    Standardized trading opportunity candidate across all symbols and styles.
    """
    symbol: str
    trade_style: str
    timeframe: str
    strategy: str
    bias: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    expected_value: float
    win_prob: float
    ml_prob: float
    confluence_score: float
    confluence_factors: List[str]
    adversarial_penalty: float
    risk_factors: List[str]
    regime: str
    regime_multiplier: float
    utility_score: float
    setup_grade: str
    is_actionable: bool
    decision_obj: Optional[Any] = None
    context: Optional[Any] = None
    market_conduction_score: float = 1.0
    profit_potential_score: float = 1.0
    dynamic_priority_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = self.decision_obj
        return {
            "symbol": self.symbol,
            "trade_style": self.trade_style,
            "timeframe": self.timeframe,
            "strategy": self.strategy,
            "bias": self.bias,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "tp1_price": getattr(d, "tp1_price", getattr(d, "first_target_price", None)) if d else None,
            "tp2_price": getattr(d, "tp2_price", self.take_profit) if d else self.take_profit,
            "tp3_price": getattr(d, "tp3_price", None) if d else None,
            "risk_reward_ratio": self.risk_reward_ratio,
            "expected_value": self.expected_value,
            "win_prob": self.win_prob,
            "ml_prob": self.ml_prob,
            "confluence_score": self.confluence_score,
            "confluence_tier": getattr(d, "master_confluence_tier", "MODERATE") if d else "MODERATE",
            "confluence_factors": self.confluence_factors,
            "adversarial_penalty": self.adversarial_penalty,
            "risk_factors": self.risk_factors,
            "regime": self.regime,
            "regime_multiplier": self.regime_multiplier,
            "utility_score": self.utility_score,
            "market_conduction_score": self.market_conduction_score,
            "profit_potential_score": self.profit_potential_score,
            "dynamic_priority_score": self.dynamic_priority_score,
            "setup_grade": self.setup_grade,
            "is_actionable": self.is_actionable,
        }

    def to_radar_item(self) -> Dict[str, Any]:
        """Converts candidate opportunity to standard HM Algo 2.0 state radar dictionary."""
        d = self.decision_obj
        ctx = self.context
        
        status_label = "NO SETUP"
        decision_str = getattr(d, "decision", "NO_TRADE") if d else "NO_TRADE"
        bias_str = self.bias

        from jarvis.market.sessions import SessionEngine
        mkt_status = SessionEngine.get_market_trading_status(self.symbol)
        is_mkt_open = mkt_status.get("is_open", True)

        if not is_mkt_open:
            status_label = "MARKET CLOSED"
        elif decision_str == "EXECUTE":
            status_label = f"{bias_str} READY"
        elif decision_str == "WAIT" and bias_str in ["BUY", "SELL"]:
            status_label = f"WAIT: {bias_str}"
        elif decision_str == "NO_TRADE":
            if d and getattr(d, "quality_gate", None) and not d.quality_gate.passed and any("Invalid" in r or "Devil" in r or "Adversarial" in r for r in getattr(d.quality_gate, "failing_reasons", [])):
                status_label = f"INVALID: {bias_str}" if bias_str in ["BUY", "SELL"] else "TRADE INVALIDATED"
            elif bias_str in ["BUY", "SELL"]:
                status_label = f"NO TRADE: {bias_str}"
            else:
                status_label = "NO SETUP"
        elif bias_str in ["BUY", "SELL"]:
            status_label = f"WAIT: {bias_str}"
        else:
            status_label = "NO SETUP"

        return {
            "symbol": self.symbol,
            "trade_style": self.trade_style,
            "timeframe": self.timeframe,
            "current_price": getattr(ctx, "current_price", self.entry_price) if ctx else self.entry_price,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "tp1_price": getattr(d, "tp1_price", getattr(d, "first_target_price", None)),
            "tp2_price": getattr(d, "tp2_price", self.take_profit),
            "tp3_price": getattr(d, "tp3_price", None),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "ev": round(self.expected_value, 2),
            "bias": self.bias,
            "action": status_label,
            "status_label": status_label,
            "decision": decision_str,
            "score": round(self.win_prob, 0),
            "win_prob": round(self.win_prob, 0),
            "ml_prob": round(self.ml_prob, 3),
            "confluence_score": round(self.confluence_score, 1),
            "confluence_tier": getattr(d, "master_confluence_tier", "MODERATE") if d else "MODERATE",
            "regime": self.regime,
            "regime_multiplier": round(self.regime_multiplier, 2),
            "strategy": self.strategy,
            "gate_passed": d.quality_gate.passed if (d and getattr(d, "quality_gate", None)) else False,
            "failing_reasons": d.quality_gate.failing_reasons if (d and getattr(d, "quality_gate", None)) else [],
            "checks": d.quality_gate.checks if (d and getattr(d, "quality_gate", None)) else {},
            "waiting_reasons": getattr(d, "waiting_reasons", []) if d else [],
            "rejection_reasons": getattr(d, "rejection_reasons", []) if d else [],
            "risk_factors": self.risk_factors,
            "adversarial_penalty": round(self.adversarial_penalty, 1),
            "invalidation_levels": getattr(d, "invalidation_levels", []) if d else [],
            "mtf_alignment": getattr(ctx, "mtf_alignment", {}) if ctx else {},
            "mtf_confluence": getattr(ctx, "mtf_confluence_score", 0.0) if ctx else 0.0,
            "utility_score": self.utility_score,
            "market_conduction_score": round(self.market_conduction_score, 2),
            "profit_potential_score": round(self.profit_potential_score, 2),
            "dynamic_priority_score": round(self.dynamic_priority_score, 4),
            "setup_grade": self.setup_grade,
            "is_actionable": self.is_actionable,
        }


class UniversalOpportunityArbiter:
    """
    Master Opportunity Arbiter.
    Ranks, arbitrates, and selects the single most mathematically superior trade setup
    across multiple asset classes (Forex, Metals, Crypto, Indices) and multiple trading horizons.
    """

    def __init__(self, ml_predictor=None, bandit=None, self_learning=None):
        self.ml_predictor = ml_predictor
        self.bandit = bandit
        self.self_learning = self_learning

    def calculate_regime_multiplier(self, regime: str, trade_style: str, strategy: str) -> float:
        """
        Computes dynamic regime compatibility multiplier.
        Rewards setups aligned with the optimal market structure phase.
        """
        reg_upper = str(regime or "RANGE").upper()
        style_upper = str(trade_style or "SWING").upper()
        strat_upper = str(strategy or "").upper()

        mult = 1.0

        # Style-Regime Matrix
        if "SWING" in style_upper:
            if "TREND_BULL" in reg_upper or "TREND_BEAR" in reg_upper:
                mult = 1.20
            elif "BREAKOUT" in reg_upper or "ACCUMULATION" in reg_upper or "DISTRIBUTION" in reg_upper:
                mult = 1.10
            elif "RANGE" in reg_upper or "CONSOLIDATION" in reg_upper:
                mult = 0.95
            elif "HIGH_VOLATILITY" in reg_upper or "EVENT_RISK" in reg_upper:
                mult = 0.80
        elif "DAY" in style_upper or "INTRADAY" in style_upper:
            if "TREND_BULL" in reg_upper or "TREND_BEAR" in reg_upper:
                mult = 1.15
            elif "BREAKOUT" in reg_upper or "POST_BREAKOUT" in reg_upper:
                mult = 1.15
            elif "RANGE" in reg_upper or "CONSOLIDATION" in reg_upper:
                mult = 1.05
            elif "HIGH_VOLATILITY" in reg_upper:
                mult = 0.90
            elif "EVENT_RISK" in reg_upper:
                mult = 0.80
        elif "SCALP" in style_upper:
            if "LIQUIDITY_SWEEP" in reg_upper or "REVERSAL" in reg_upper:
                mult = 1.25
            elif "RANGE" in reg_upper or "COMPRESSION" in reg_upper:
                mult = 1.15
            elif "HIGH_VOLATILITY" in reg_upper:
                mult = 1.00
            elif "EVENT_RISK" in reg_upper or "LIQUIDITY_STRESS" in reg_upper:
                mult = 0.75

        # Strategy-Regime Synergy Bonus
        if "SWEEP" in strat_upper and "LIQUIDITY" in reg_upper:
            mult += 0.05
        elif "TREND" in strat_upper and "TREND" in reg_upper:
            mult += 0.05
        elif "RANGE" in strat_upper and "RANGE" in reg_upper:
            mult += 0.05

        # Query self-learning historical regime adjustment if available
        if self.self_learning and hasattr(self.self_learning, "get_regime_multiplier"):
            try:
                hist_mult = self.self_learning.get_regime_multiplier(reg_upper)
                mult = (mult * 0.7) + (hist_mult * 0.3)
            except Exception as e:
                logger.debug(f"Self-learning regime multiplier unavailable for {reg_upper}: {e}")

        return round(float(mult), 3)

    def calculate_market_conduction(
        self,
        symbol: str,
        context: Optional[MarketContext],
        decision_obj: Optional[DecisionObject] = None,
        style: str = "SWING"
    ) -> float:
        """
        Evaluates dynamic market conduction: how cleanly and efficiently the market
        structure transmits price movement toward profit targets.
        Factors:
        - Trend velocity & strength (ADX >= 25)
        - Volatility expansion vs compression/extreme
        - Spread-to-risk friction impedance
        - Optimal Pricing Zone (OTE discount for BUY, premium for SELL)
        - Peak session volume liquidity
        - Intermarket / SMT confirmation
        Returns a conduction multiplier in [0.50, 1.50], defaulting to 1.0 when context is absent.
        """
        if context is None:
            return 1.0

        conduction = 1.0
        bias = str(getattr(decision_obj, "bias", "") or (getattr(context, "structure", None) and getattr(context.structure, "bias", "")) or "").upper()

        # 1. Trend Velocity & Momentum Conduction (ADX & Trend Score)
        momentum = getattr(context, "momentum", None)
        if momentum is not None:
            adx_val = float(getattr(momentum, "adx", 0.0) or 0.0)
            trend_sc = float(getattr(momentum, "trend_score", 0.0) or 0.0)

            if adx_val >= 30.0:
                conduction += 0.15  # Pristine institutional trending velocity
            elif adx_val >= 22.0:
                conduction += 0.08  # Healthy directional conduction
            elif 0.0 < adx_val < 18.0:
                conduction -= 0.10  # Sluggish chop / consolidation drag

            # Directional alignment bonus
            if bias == "BUY" and trend_sc >= 35.0:
                conduction += 0.08
            elif bias == "SELL" and trend_sc <= -35.0:
                conduction += 0.08
            elif (bias == "BUY" and trend_sc < -20.0) or (bias == "SELL" and trend_sc > 20.0):
                conduction -= 0.15  # Counter-momentum friction

        # 2. Volatility State & Expansion Conduction
        volatility = getattr(context, "volatility", None)
        if volatility is not None:
            vol_state = str(getattr(volatility, "state", "NORMAL")).upper()
            if vol_state == "EXPANSION":
                conduction += 0.10  # Volatility expansion rapidly conducts price to targets
            elif vol_state == "NORMAL":
                conduction += 0.05
            elif vol_state == "COMPRESSION":
                conduction -= 0.05  # Energy coiling, slower initial conduction
            elif vol_state in ("EXTREME", "HIGH_VOLATILITY"):
                conduction -= 0.18  # Hazardous whipsaw risk

            if getattr(volatility, "volatility_spike_breaker", False):
                conduction -= 0.25

            # 3. Spread-to-Risk Friction Impedance
            cur_spread = float(getattr(volatility, "current_spread_pips", 0.0) or 0.0)
            entry_p = float(getattr(decision_obj, "entry_price", 0.0) or 0.0) if decision_obj else 0.0
            sl_p = float(getattr(decision_obj, "stop_loss", 0.0) or 0.0) if decision_obj else 0.0
            risk_dist = abs(entry_p - sl_p)
            if cur_spread > 0 and risk_dist > 1e-9:
                from jarvis.data.symbol_registry import resolve as resolve_symbol
                try:
                    spec = resolve_symbol(symbol)
                    pip_sz = getattr(spec, "pip_size", 0.0001) or 0.0001
                    friction_ratio = (cur_spread * pip_sz) / risk_dist
                    if friction_ratio <= 0.06:
                        conduction += 0.10  # Ultra-clean price conduction: broker spread takes <6% of risk
                    elif friction_ratio <= 0.12:
                        conduction += 0.05
                    elif friction_ratio >= 0.30:
                        conduction -= 0.20  # High drag: spread consumes >30% of stop distance
                except Exception:
                    pass

        # 4. Optimal Pricing Zone (OTE) Structural Conduction
        structure = getattr(context, "structure", None)
        if structure is not None:
            zone = str(getattr(structure, "discount_premium_zone", "")).upper()
            if bias == "BUY" and ("DISCOUNT" in zone or "EQUILIBRIUM" in zone):
                conduction += 0.10  # Buying low allows full structural expansion upward
            elif bias == "SELL" and ("PREMIUM" in zone or "EQUILIBRIUM" in zone):
                conduction += 0.10  # Selling high allows full structural expansion downward
            elif (bias == "BUY" and "PREMIUM" in zone) or (bias == "SELL" and "DISCOUNT" in zone):
                conduction -= 0.15  # Buying expensive or selling cheap into structural barrier

        # 5. Session Volume & Institutional Liquidity Conduction
        session = getattr(context, "session", None)
        if session is not None:
            is_prime = getattr(session, "is_prime_session", False)
            if is_prime:
                conduction += 0.10  # Deep institutional order book conducts price cleanly
            else:
                is_crypto = any(c in symbol.upper() for c in ("BTC", "ETH", "SOL", "CRYPTO"))
                if not is_crypto:
                    conduction -= 0.08  # Off-session thin liquidity drag

        # 6. Intermarket & SMT Macro Confluence
        dxy_info = getattr(context, "intermarket_dxy", {}) or {}
        if isinstance(dxy_info, dict) and dxy_info.get("is_live"):
            gold_asset = any(g in symbol.upper() for g in ("XAU", "GOLD"))
            if gold_asset and bias == "BUY" and "CONFIRMS_BUY" in str(dxy_info.get("gold_impact", "")).upper():
                conduction += 0.08
            elif gold_asset and bias == "SELL" and "CONFIRMS_SELL" in str(dxy_info.get("gold_impact", "")).upper():
                conduction += 0.08

        return round(float(np.clip(conduction, 0.50, 1.50)), 3)

    def calculate_profit_potential(
        self,
        expected_value: float,
        rr_ratio: float,
        win_prob: float,
        symbol: str,
        context: Optional[MarketContext] = None,
        decision_obj: Optional[DecisionObject] = None
    ) -> float:
        """
        Quantifies dynamic profit potential based on:
        - Expected Value in R (EV_R) and dollar expectancy (EV_$)
        - Structural risk/reward asymmetry (RR >= 2.0 to 4.0)
        - Model win probability
        - ATR expansion capacity
        Returns a profit potential multiplier in [0.50, 2.50], defaulting to 1.0 when context is absent.
        """
        ev_clamped = max(0.0, float(expected_value or 0.0))
        rr_clamped = max(1.0, float(rr_ratio or 1.5))

        # 1. Expected Value Scaling
        if ev_clamped >= 2.0:
            ev_factor = 1.35
        elif ev_clamped >= 1.2:
            ev_factor = 1.20
        elif ev_clamped >= 0.8:
            ev_factor = 1.10
        elif ev_clamped >= 0.4:
            ev_factor = 1.00
        else:
            ev_factor = 0.85

        # 2. Risk/Reward Asymmetry Factor
        if rr_clamped >= 3.5:
            rr_factor = 1.25
        elif rr_clamped >= 2.5:
            rr_factor = 1.15
        elif rr_clamped >= 2.0:
            rr_factor = 1.05
        else:
            rr_factor = 0.90

        # 3. Dollar Expectancy Check
        dollar_ev = float(getattr(decision_obj, "expected_value_dollars", 0.0) or 0.0) if decision_obj else 0.0
        dollar_boost = 1.0
        if dollar_ev > 10.0:
            dollar_boost = 1.10
        elif dollar_ev > 5.0:
            dollar_boost = 1.05
        elif dollar_ev < 0.0:
            dollar_boost = 0.85

        # 4. Volatility Expansion Capacity (ATR / Price ratio)
        atr_boost = 1.0
        if context is not None:
            vol = getattr(context, "volatility", None)
            cur_p = getattr(context, "current_price", 0.0) or 1.0
            if vol is not None and getattr(vol, "atr", 0.0) > 0 and cur_p > 0:
                atr_pct = (vol.atr / cur_p) * 100.0
                if atr_pct >= 2.0:
                    atr_boost = 1.10  # High expansion capacity (e.g. Crypto/Metals in expansion)
                elif atr_pct >= 0.8:
                    atr_boost = 1.05

        profit_score = ev_factor * rr_factor * dollar_boost * atr_boost
        return round(float(np.clip(profit_score, 0.50, 2.50)), 3)

    def evaluate_opportunity(
        self,
        decision_obj: DecisionObject,
        context: Optional[MarketContext] = None,
        trade_style: str = "SWING"
    ) -> CandidateOpportunity:
        """
        Evaluates a single decision object through the Universal Opportunity Arbiter.
        Calculates Utility = P_ML * E[V] * (1 + Confluence / 100) * (1 - Penalty) * RegimeMultiplier
        and assigns setup grades (GRADE A+, GRADE A, GRADE B, GRADE C).
        """
        symbol = getattr(decision_obj, "symbol", "UNKNOWN")
        strategy = getattr(decision_obj, "strategy", "STRUCTURE") or "STRUCTURE"
        bias = getattr(decision_obj, "bias", "HOLD")
        entry_price = float(getattr(decision_obj, "entry_price", 0.0) or 0.0)
        stop_loss = float(getattr(decision_obj, "stop_loss", 0.0) or 0.0)
        take_profit = float(getattr(decision_obj, "take_profit", 0.0) or 0.0)
        rr_ratio = float(getattr(decision_obj, "risk_reward_ratio", 0.0) or 0.0)
        expected_value = float(getattr(decision_obj, "expected_value", 0.0) or 0.0)

        # Timeframe mapping
        style_norm = str(trade_style or "SWING").upper()
        if "D1" in style_norm or "DAILY" in style_norm:
            timeframe = "W1/D1/H4"
        elif "H4" in style_norm:
            timeframe = "D1/H4/H1"
        elif "SWING" in style_norm:
            timeframe = "D1/H4/H1"
        elif "DAY" in style_norm or "INTRADAY" in style_norm:
            timeframe = "H1/M15/M5"
        else:
            timeframe = "M15/M5/M1"

        # Win probability (percentage 0.0-100.0 and probability 0.0-1.0)
        raw_prob = getattr(decision_obj, "model_confidence", 0.50) or 0.50
        if hasattr(decision_obj, "probabilities") and decision_obj.probabilities and bias in ("BUY", "SELL"):
            raw_prob = decision_obj.probabilities.get(bias.lower(), raw_prob)

        win_prob_pct = (raw_prob * 100.0) if raw_prob <= 1.0 else raw_prob
        win_prob_decimal = win_prob_pct / 100.0

        # ML Probability (P_ML)
        ml_prob = getattr(decision_obj, "meta_label_prob", None)
        if ml_prob is None or ml_prob <= 0:
            if self.ml_predictor and context is not None:
                try:
                    reg_obj = getattr(decision_obj, "regime", None)
                    features = self.ml_predictor.extract_features(
                        context=context,
                        regime=reg_obj,
                        trade_style=style_norm,
                        strategy=strategy,
                        tentative_bias=bias,
                        devil_penalty=getattr(decision_obj, "adversarial_penalty", 0.0),
                        target_rr=rr_ratio
                    )
                    ml_prob = self.ml_predictor.predict_probability(features)
                except Exception as e:
                    logger.debug(f"Arbiter ML feature extraction fallback: {e}")
                    ml_prob = win_prob_decimal
            else:
                ml_prob = win_prob_decimal

        ml_prob = float(np.clip(ml_prob, 0.35, 0.88))

        # Master Confluence Score (0.0 to 100.0)
        confluence_score = float(getattr(decision_obj, "master_confluence_score", 0.0) or 0.0)
        if confluence_score == 0.0 and context is not None:
            confluence_score = float(getattr(context, "mtf_confluence_score", 50.0) or 50.0)

        confluence_factors = []
        if getattr(decision_obj, "bull_case", None):
            confluence_factors.extend(decision_obj.bull_case[:3])
        if getattr(decision_obj, "bear_case", None) and bias == "SELL":
            confluence_factors.extend(decision_obj.bear_case[:3])

        risk_factors = list(getattr(decision_obj, "risk_factors", []) or [])

        # Adversarial Penalty (normalized 0.0 to 1.0)
        raw_penalty = float(getattr(decision_obj, "adversarial_penalty", 0.0) or 0.0)
        penalty_norm = min(0.60, max(0.0, raw_penalty / 100.0 if raw_penalty > 1.0 else raw_penalty))

        # Regime & Regime Multiplier
        reg_obj = getattr(decision_obj, "regime", None)
        if reg_obj and hasattr(reg_obj, "primary_regime"):
            regime_str = reg_obj.primary_regime.value if hasattr(reg_obj.primary_regime, "value") else str(reg_obj.primary_regime)
        else:
            regime_str = str(reg_obj or "RANGE")

        regime_multiplier = self.calculate_regime_multiplier(regime_str, style_norm, strategy)

        # Mathematical Utility Computation:
        # Utility = P_ML * E[V] * (1 + Confluence / 100) * (1 - Penalty) * RegimeMultiplier
        effective_ev = max(0.0, expected_value)
        if effective_ev > 5.0:
            effective_ev = min(3.5, round(effective_ev / 20.0, 2))
        confluence_term = 1.0 + (confluence_score / 100.0)
        penalty_term = max(0.20, 1.0 - penalty_norm)

        if expected_value <= 0.0 or bias not in ("BUY", "SELL"):
            utility_score = 0.0
        else:
            utility_score = round(
                float(ml_prob * effective_ev * confluence_term * penalty_term * regime_multiplier),
                4
            )

        # Strict Regime-to-Strategy Incompatibility Gating
        from jarvis.intelligence.regime_engine import validate_strategy_for_regime
        adx_val = float(getattr(getattr(context, "momentum", None), "adx", 0.0) or 0.0) if context else 0.0
        strat_compat, compat_reason = validate_strategy_for_regime(strategy, regime_str, adx=adx_val)
        if not strat_compat:
            utility_score = 0.0

        # Spread-to-Risk Friction Filter: Mode-adaptive friction handling (only evaluated on directional setups)
        if bias in ("BUY", "SELL") and context is not None and abs(entry_price - stop_loss) > 1e-9:
            from jarvis.data.symbol_registry import resolve as resolve_symbol
            try:
                sym_spec = resolve_symbol(symbol)
                pip_sz = getattr(sym_spec, "pip_size", 0.0001) or 0.0001
                spread_pips_val = float(getattr(getattr(context, "volatility", None), "current_spread_pips", 0.0) or 0.0)
                spread_dist_val = spread_pips_val * pip_sz
                risk_dist_val = abs(entry_price - stop_loss)
                friction_ratio = spread_dist_val / risk_dist_val
                is_crypto = any(c in symbol.upper() for c in ("BTC", "ETH", "SOL", "CRYPTO"))
                # SCALP and Crypto naturally encounter higher friction on tight stops
                max_friction = 0.35 if (style_norm == "SCALP" or is_crypto) else (0.25 if style_norm == "DAY_TRADING" else 0.18)
                if friction_ratio > max_friction:
                    if friction_ratio > 0.50:  # Excessive: spread consumes >50% of the stop loss
                        logger.info(f"🛡️ Arbiter: Disqualifying {symbol} setup: spread eats {friction_ratio*100:.1f}% (>50%) of risk distance.")
                        utility_score = 0.0
                    else:  # Moderate friction: apply proportional penalty rather than killing the trade outright
                        penalty_mult = max(0.20, 1.0 - (friction_ratio - max_friction) * 2.0)
                        utility_score = round(utility_score * penalty_mult, 4)
                        logger.debug(f"🛡️ Arbiter: Penalizing {symbol} utility by {penalty_mult:.2f}x: friction {friction_ratio*100:.1f}% > {max_friction*100:.0f}%")
            except Exception as e:
                logger.debug(f"Arbiter spread friction check: {e}")

        # Session Quality Filter: Prohibit new entries in OFF_HOURS on non-crypto assets
        from jarvis.market.sessions import SessionEngine
        try:
            sess_info = SessionEngine.get_current_session()
            is_crypto_asset = any(c in symbol.upper() for c in ("BTC", "ETH", "SOL", "CRYPTO"))
            from jarvis.intelligence.symbol_profile_config import get_symbol_profile_config
            sym_prof_cfg = get_symbol_profile_config(symbol)
            has_sess_rest = getattr(sym_prof_cfg, "session_restriction", True)
            if sess_info.current_session == "OFF_HOURS" and not is_crypto_asset and has_sess_rest:
                logger.debug(f"🛡️ Arbiter: OFF_HOURS session active -- suppressing new non-crypto entry for {symbol}")
                utility_score = 0.0
        except Exception as e:
            logger.debug(f"Arbiter session check: {e}")

        # Multi-Timeframe Alignment Check: Penalize counter-macro trades
        mtf_data = getattr(context, "mtf_alignment", {}) if context else {}
        is_counter_flush_kill = False
        if isinstance(mtf_data, dict):
            macro_w1 = str(mtf_data.get("W1", "")).upper()
            macro_d1 = str(mtf_data.get("D1", "")).upper()
            macro_h1_bias = str(mtf_data.get("H1", "")).upper()
            macro_h4 = str(mtf_data.get("H4", "")).upper()
            m15_bias = str(mtf_data.get("M15", "")).upper()
            m5_bias = str(mtf_data.get("M5", "")).upper()

            if bias == "BUY" and ("BEAR" in macro_w1 or "BEAR" in macro_d1 or "BEAR" in macro_h4):
                utility_score = round(utility_score * 0.65, 4)
            elif bias == "SELL" and ("BULL" in macro_w1 or "BULL" in macro_d1 or "BULL" in macro_h4):
                utility_score = round(utility_score * 0.65, 4)

            # Institutional Rule R2: Never short into active M15/M5 upward expansion or buy into active downward flush
            counter_flush_active = False
            if bias == "SELL" and ("BULL" in m15_bias and "BULL" in m5_bias):
                counter_flush_active = True
                utility_score = round(utility_score * 0.45, 4)
                logger.info(f"🛡️ Arbiter: Severely penalizing {symbol} SELL by 0.45x: M15 and M5 are actively BULLISH.")
            elif bias == "BUY" and ("BEAR" in m15_bias and "BEAR" in m5_bias):
                counter_flush_active = True
                utility_score = round(utility_score * 0.45, 4)
                logger.info(f"🛡️ Arbiter: Severely penalizing {symbol} BUY by 0.45x: M15 and M5 are actively BEARISH.")

            # Counter-flush signature is disqualifying across ALL symbols (Gold, Crypto, Indices, FX).
            # An entry against M15+M5 while H1, H4 or D1 runs the other way is a fatal counter-trend trade.
            if counter_flush_active:
                stack_confirms = (
                    ("BEAR" in macro_h1_bias and bias == "BUY")
                    or ("BEAR" in macro_d1 and bias == "BUY")
                    or ("BEAR" in macro_h4 and bias == "BUY")
                    or ("BULL" in macro_h1_bias and bias == "SELL")
                    or ("BULL" in macro_d1 and bias == "SELL")
                    or ("BULL" in macro_h4 and bias == "SELL")
                )
                if stack_confirms:
                    is_counter_flush_kill = True
                    logger.info(
                        f"🛑 Arbiter: {symbol} {bias} candidate DISQUALIFIED — counter-flush into an "
                        f"agreed HTF trend (D1={macro_d1 or 'NA'}, H4={macro_h4 or 'NA'}, H1={macro_h1_bias or 'NA'}, M15={m15_bias}, M5={m5_bias})."
                    )
        # Intermarket Macro Intelligence & SMT Cross-Market Multipliers
        if context is not None and utility_score > 0.0:
            # 1. Synthetic Dollar Index (DXY) alignment
            dxy_info = getattr(context, "intermarket_dxy", {}) or {}
            if isinstance(dxy_info, dict) and dxy_info.get("is_live"):
                is_gold_asset = any(g in symbol.upper() for g in ("XAU", "GOLD"))
                if is_gold_asset:
                    g_impact = str(dxy_info.get("gold_impact", "")).upper()
                    if bias == "BUY":
                        if "CONFIRMS_BUY" in g_impact:
                            utility_score = round(utility_score * 1.10, 4)
                            confluence_factors.append(f"DXY confirms Gold BUY (DXY {dxy_info.get('trend')})")
                        elif "OPPOSES_BUY" in g_impact:
                            utility_score = round(utility_score * 0.85, 4)
                            risk_factors.append(f"DXY opposes Gold BUY (DXY {dxy_info.get('trend')} momentum {dxy_info.get('momentum_score')})")
                    elif bias == "SELL":
                        if "CONFIRMS_SELL" in g_impact:
                            utility_score = round(utility_score * 1.10, 4)
                            confluence_factors.append(f"DXY confirms Gold SELL (DXY {dxy_info.get('trend')})")
                        elif "OPPOSES_SELL" in g_impact:
                            utility_score = round(utility_score * 0.85, 4)
                            risk_factors.append(f"DXY opposes Gold SELL (DXY {dxy_info.get('trend')} momentum {dxy_info.get('momentum_score')})")
                else:
                    fx_bias_map = dxy_info.get("fx_bias", {})
                    if isinstance(fx_bias_map, dict) and symbol in fx_bias_map:
                        target_fx_bias = fx_bias_map[symbol]
                        if target_fx_bias == bias:
                            utility_score = round(utility_score * 1.08, 4)
                            confluence_factors.append(f"DXY macro trend aligns with {symbol} {bias}")
                        elif target_fx_bias in ("BUY", "SELL") and target_fx_bias != bias:
                            utility_score = round(utility_score * 0.88, 4)
                            risk_factors.append(f"DXY macro trend opposes {symbol} {bias}")

            # 2. Cross-Market SMT Divergence alignment
            smt_info = getattr(context, "cross_market_smt", {}) or getattr(context, "metals_smt", {}) or {}
            if isinstance(smt_info, dict) and smt_info.get("smt_active"):
                smt_type = str(smt_info.get("smt_type", "")).upper()
                leader = smt_info.get("leader", "peer")
                if "BULLISH" in smt_type:
                    if bias == "BUY":
                        utility_score = round(utility_score * 1.12, 4)
                        confluence_factors.append(f"SMT Bullish Divergence accumulation ({leader} leading)")
                    elif bias == "SELL":
                        utility_score = round(utility_score * 0.80, 4)
                        risk_factors.append(f"Opposing SMT Bullish Divergence active ({leader} leading)")
                elif "BEARISH" in smt_type:
                    if bias == "SELL":
                        utility_score = round(utility_score * 1.12, 4)
                        confluence_factors.append(f"SMT Bearish Divergence distribution ({leader} leading)")
                    elif bias == "BUY":
                        utility_score = round(utility_score * 0.80, 4)
                        risk_factors.append(f"Opposing SMT Bearish Divergence active ({leader} leading)")

        # Setup Grade Assignment:
        # Dynamic style-aware utility tiers: Scalp and Day Trading naturally produce smaller
        # EV (0.3R to 0.9R) than multi-day Swing (1.2R to 2.5R). Calibrate tiers per style:
        if style_norm == "SCALP":
            if utility_score >= 0.85 and (win_prob_pct >= 60.0 or ml_prob >= 0.65) and confluence_score >= 25.0 and expected_value >= 0.50:
                setup_grade = "GRADE A+"
            elif utility_score >= 0.45 and expected_value >= 0.25:
                setup_grade = "GRADE A"
            elif utility_score >= 0.20 and expected_value > 0:
                setup_grade = "GRADE B"
            else:
                setup_grade = "GRADE C"
            min_actionable_utility = 0.20
        elif style_norm == "DAY_TRADING":
            if utility_score >= 1.20 and (win_prob_pct >= 60.0 or ml_prob >= 0.65) and confluence_score >= 25.0 and expected_value >= 0.80:
                setup_grade = "GRADE A+"
            elif utility_score >= 0.60 and expected_value >= 0.35:
                setup_grade = "GRADE A"
            elif utility_score >= 0.20 and expected_value > 0:
                setup_grade = "GRADE B"
            else:
                setup_grade = "GRADE C"
            min_actionable_utility = 0.20
        else:
            # SWING / Default:
            if utility_score >= 1.80 and (win_prob_pct >= 62.0 or ml_prob >= 0.70) and confluence_score >= 28.0 and expected_value >= 0.80:
                setup_grade = "GRADE A+"
            elif utility_score >= 1.30 and (win_prob_pct >= 55.0 or ml_prob >= 0.60) and confluence_score >= 22.0 and expected_value >= 0.40:
                setup_grade = "GRADE A"
            elif utility_score >= 0.50 and expected_value > 0:
                setup_grade = "GRADE B"
            else:
                setup_grade = "GRADE C"
            min_actionable_utility = 0.50

        # Actionability Determination: Require Grade A/A+ or solid Grade B with positive EV and passed/acceptable quality gate
        failing_reasons = getattr(decision_obj.quality_gate, "failing_reasons", []) if (decision_obj and getattr(decision_obj, "quality_gate", None)) else []
        softened = {
            canonical_gate(g)
            for g in (getattr(decision_obj.quality_gate, "softened_gates", []) if decision_obj else [])
        }
        canonical_failures = [canonical_gate(r) for r in failing_reasons]
        unsoftened_failures = [r for r in canonical_failures if r not in softened]
        critical_names = {
            "Drawdown Safety Guard",
            "Spread Protection",
            "Regime Viability",
            "Volatility Spike Circuit Breaker",
            "Market Session Open",
            "Margin Capacity Limit",
        }
        critical_failures = [r for r in unsoftened_failures if r in critical_names]
        # A candidate with one failed gate used to stay actionable as long as the
        # failure was not in `critical_failures` - measured on jarvis_history.db:
        # THAT is how banned-strategy gold trades executed on 2026-10-13..16
        # ("Strategy Viable" being the single failing reason) and how 00-06h
        # gold trades fired under a closed session window ("Forex Prime
        # Session" being the single failing reason). A HARD gate failure is
        # never an acceptable residual: consult the same HARD_GATES set the
        # adaptive gate policy enforces at decision time, so the two layers
        # cannot disagree about what may still trade.
        hard_failures = [r for r in unsoftened_failures if r in HARD_GATES]
        effective_passed = getattr(decision_obj.quality_gate, "effective_passed", None) if decision_obj else None
        if effective_passed is None:
            effective_passed = bool(getattr(decision_obj.quality_gate, "passed", False)) if decision_obj else False
        gate_acceptable = (
            bool(effective_passed)
            or (
                len(set(unsoftened_failures)) <= 1
                and len(critical_failures) == 0
                and len(hard_failures) == 0
            )
        )
        decision_val = getattr(decision_obj, "decision", "")

        is_actionable = bool(
            bias in ("BUY", "SELL") and
            utility_score >= min_actionable_utility and
            expected_value > 0 and
            gate_acceptable and
            setup_grade in ("GRADE A+", "GRADE A", "GRADE B") and
            decision_val in ("EXECUTE", "READY", "WAIT") and
            not is_counter_flush_kill
        )

        # Nano account risk ceiling sanity check ($50 - $250 equity)
        from jarvis.risk.account_tier import is_nano_account, get_nano_risk_cap
        from jarvis.data.symbol_registry import get_dollar_risk_per_price_unit
        acct_bal = float(getattr(decision_obj, "account_balance", 0.0) or (getattr(context, "account_equity", 0.0) if context else 0.0) or 0.0)
        if is_actionable and acct_bal > 0 and is_nano_account(acct_bal):
            max_risk_usd, _ = get_nano_risk_cap(acct_bal, symbol)
            mpu = float(get_dollar_risk_per_price_unit(symbol) or 0.0)
            planned_risk_usd = abs(entry_price - stop_loss) * mpu * 0.01  # minimum 0.01 lot
            if planned_risk_usd > max_risk_usd + 1e-9:
                is_actionable = False
                risk_factors.append(f"Exceeds nano account risk ceiling (${planned_risk_usd:.2f} > ${max_risk_usd:.2f})")
                logger.info(
                    f"🛡️ Arbiter: Disqualifying {symbol} setup for nano account (${acct_bal:.1f}): "
                    f"0.01 lot planned risk ${planned_risk_usd:.2f} exceeds nano ceiling ${max_risk_usd:.2f}"
                )

        # Dynamic Market Conduction & High-Profit Priority Analysis
        market_conduction_score = self.calculate_market_conduction(
            symbol=symbol,
            context=context,
            decision_obj=decision_obj,
            style=style_norm
        )
        profit_potential_score = self.calculate_profit_potential(
            expected_value=effective_ev,
            rr_ratio=rr_ratio,
            win_prob=win_prob_pct,
            symbol=symbol,
            context=context,
            decision_obj=decision_obj
        )
        dynamic_priority_score = round(
            float(utility_score * market_conduction_score * profit_potential_score),
            4
        )

        return CandidateOpportunity(
            symbol=symbol,
            trade_style=style_norm,
            timeframe=timeframe,
            strategy=strategy,
            bias=bias,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr_ratio,
            expected_value=expected_value,
            win_prob=win_prob_pct,
            ml_prob=ml_prob,
            confluence_score=confluence_score,
            confluence_factors=confluence_factors,
            adversarial_penalty=raw_penalty,
            risk_factors=risk_factors,
            regime=regime_str,
            regime_multiplier=regime_multiplier,
            utility_score=utility_score,
            setup_grade=setup_grade,
            is_actionable=is_actionable,
            decision_obj=decision_obj,
            context=context,
            market_conduction_score=market_conduction_score,
            profit_potential_score=profit_potential_score,
            dynamic_priority_score=dynamic_priority_score,
        )

    def rank_and_select_best(
        self,
        candidates: List[CandidateOpportunity]
    ) -> Tuple[Optional[CandidateOpportunity], List[CandidateOpportunity]]:
        """
        Ranks all candidate opportunities by dynamic profit priority and utility score descending.
        Selects the best actionable Grade A/A+ opportunity.
        Returns (best_actionable_opportunity, ranked_list).
        """
        if not candidates:
            return None, []

        # Dynamic Market Conduction & Profit Priority Ranking:
        # Candidates with elite dynamic market conduction and superior profit expectancy
        # are hoisted to Super-Priority Tier 0 regardless of asset class.
        # Within each priority tier, candidates are ordered by dynamic priority score,
        # utility, win probability, confluence, and expected value.
        # When dynamic context is absent (e.g. synthetic test fixtures), the ranking
        # cleanly falls back to the baseline profile priority (XAUUSD -> BTC/ETH -> FX/Indices).
        def _priority_bucket(c: CandidateOpportunity) -> int:
            sym_upper = str(getattr(c, "symbol", "")).upper()
            is_act = getattr(c, "is_actionable", False)
            grade = getattr(c, "setup_grade", "")
            if is_act and grade in ("GRADE A+", "GRADE A"):
                dyn_prio = float(getattr(c, "dynamic_priority_score", 0.0) or 0.0)
                mkt_cond = float(getattr(c, "market_conduction_score", 1.0) or 1.0)
                ev_val = float(getattr(c, "expected_value", 0.0) or 0.0)

                # ELITE DYNAMIC PROFIT OPPORTUNITY (Super-Priority Tier 0):
                # When dynamic analysis confirms exceptional market conduction (>= 1.05),
                # robust expected value (>= 0.80R), and high dynamic priority score (>= 1.50),
                # hoist the candidate to Super-Priority Bucket 0 regardless of asset class!
                if dyn_prio >= 1.50 and mkt_cond >= 1.05 and ev_val >= 0.80:
                    return 0

                # Baseline profile fallback when dynamic context is absent or standard:
                if "XAU" in sym_upper or "GOLD" in sym_upper:
                    return 0
                if "BTC" in sym_upper or "ETH" in sym_upper:
                    return 1
                return 2
            return 3

        ranked = sorted(
            candidates,
            key=lambda c: (
                _priority_bucket(c),
                -getattr(c, "dynamic_priority_score", c.utility_score),
                -c.utility_score,
                -c.win_prob,
                -c.confluence_score,
                -c.expected_value
            )
        )

        best_actionable = None
        for cand in ranked:
            if cand.is_actionable and cand.setup_grade in ("GRADE A+", "GRADE A"):
                best_actionable = cand
                break

        if not best_actionable:
            for cand in ranked:
                if cand.is_actionable and cand.setup_grade == "GRADE B":
                    best_actionable = cand
                    break

        return best_actionable, ranked
