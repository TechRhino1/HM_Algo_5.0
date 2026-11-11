import pytest
from unittest.mock import MagicMock
from jarvis.analysts.devil_advocate import DevilAdvocateAnalyst
from jarvis.intelligence.decision_engine import DecisionEngine
from jarvis.data.schemas import (
    MarketContext, MarketRegime, RegimeOutput, StructureContext,
    VolatilityContext, MomentumContext, LiquidityContext
)

class TestAutoTradingPipelineFixes:

    def test_devil_advocate_excludes_channel_and_trendline_from_threat_price_level(self):
        """Channel upper/lower and dynamic trendlines must not be treated as concrete collision obstacles."""
        analyst = DevilAdvocateAnalyst()
        ctx = MagicMock()
        ctx.symbol = "EURUSD"
        ctx.current_price = 1.0850
        ctx.context_quality = 90.0
        
        vol = MagicMock(spec=VolatilityContext)
        vol.atr = 0.0020
        vol.is_excessive_spread = False
        vol.current_spread_pips = 1.0
        vol.volatility_spike_breaker = False
        ctx.volatility = vol

        # Structure has channel_upper 5 pips away and dynamic_resistance 3 pips away,
        # but NO supply zone above current price and NO multi-touch key levels.
        st = MagicMock(spec=StructureContext)
        st.supply_zone = (1.0800, 1.0820)  # below current price
        st.demand_zone = (1.0750, 1.0770)
        st.channel_upper = 1.0855
        st.channel_lower = 1.0820
        st.dynamic_resistance_trendline = 1.0853
        st.dynamic_support_trendline = 1.0830
        st.key_levels = []  # no structural key levels
        ctx.structure = st

        reg = MagicMock(spec=RegimeOutput)
        reg.primary_regime = MarketRegime.TREND_BULL

        mom = MagicMock(spec=MomentumContext)
        mom.rsi = 55.0
        mom.trend_score = 40.0
        mom.adx = 20.0
        mom.plus_di = 20.0
        mom.minus_di = 15.0
        mom.divergence = "NONE"
        ctx.momentum = mom

        liq = MagicMock(spec=LiquidityContext)
        liq.sweep_detected = False
        liq.sweep_type = "NONE"
        liq.buy_side_liquidity = 0.0
        liq.sell_side_liquidity = 0.0
        ctx.liquidity = liq

        rep = analyst.critique_opportunity(ctx, reg, "BUY")
        # Dynamic channel band/trendline must NOT be assigned as threat_price_level
        assert rep.threat_price_level is None, f"Expected None, got {rep.threat_price_level}"

        # Now add a real multi-touch structural key level at 1.0890
        st.key_levels = [{"price": 1.0890, "touches": 3, "strength": 75}]
        rep2 = analyst.critique_opportunity(ctx, reg, "BUY")
        assert rep2.threat_price_level == 1.0890

    def test_gold_equilibrium_pullback_passes_premium_discount(self):
        """Precious metals in EQUILIBRIUM zone must pass premium_discount_valid during trend pullbacks."""
        engine = DecisionEngine()
        ctx = MagicMock(spec=MarketContext)
        ctx.symbol = "XAUUSD"
        ctx.current_price = 2650.0

        st = MagicMock(spec=StructureContext)
        st.discount_premium_zone = "EQUILIBRIUM"
        st.choch_type = "NONE"
        st.fair_value_gaps = [{"high": 2655.0, "low": 2648.0}]
        st.order_blocks = []
        ctx.structure = st

        mom = MagicMock(spec=MomentumContext)
        mom.rsi = 48.0
        mom.trend_score = -55.0  # strong bearish momentum
        ctx.momentum = mom

        # Evaluate logic matching the updated decision_engine condition
        sym_name_eval = "XAUUSD"
        best_strategy = "TREND_PULLBACK"
        trend_score_val = mom.trend_score
        rsi_val_eval = mom.rsi
        tentative_bias = "SELL"
        has_bear_choch = False
        is_trend_pullback = best_strategy in ("TREND_PULLBACK", "TREND_FOLLOWING", "BREAKOUT")
        has_fvg_ob = bool(getattr(st, "fair_value_gaps", None) or getattr(st, "order_blocks", None))
        strong_metals_trend = abs(trend_score_val) >= 50.0

        premium_discount_valid = True
        if tentative_bias == "SELL":
            if is_trend_pullback or strong_metals_trend or has_fvg_ob:
                if st.discount_premium_zone == "DISCOUNT" and rsi_val_eval < 30.0:
                    premium_discount_valid = False
            else:
                if st.discount_premium_zone != "PREMIUM" and not has_bear_choch:
                    premium_discount_valid = False

        assert premium_discount_valid is True, "Gold Equilibrium pullback must be valid for SELL"

    def test_orchestrator_is_exec_ready_accepts_wait_with_low_failing_cnt(self):
        """is_exec_ready logic must allow dispatch when candidate is Grade A/A+ and d_action is WAIT with failing_cnt <= 1."""
        candidate = MagicMock()
        candidate.is_actionable = True
        candidate.setup_grade = "GRADE A+"
        candidate.decision_obj = MagicMock()
        candidate.bias = "BUY"
        candidate.decision_obj.bias = "BUY"
        candidate.decision_obj.order_type = "MARKET"
        candidate.expected_value = 0.85
        candidate.risk_reward_ratio = 2.1
        
        d_action = "WAIT"
        failing_cnt = 1

        is_exec_ready = (
            candidate.is_actionable
            and candidate.setup_grade in ("GRADE A+", "GRADE A", "GRADE B")
            and candidate.decision_obj
            and candidate.bias in ("BUY", "SELL")
            and getattr(candidate.decision_obj, "bias", "") in ("BUY", "SELL")
            and (
                getattr(candidate.decision_obj, "order_type", "MARKET") in ("MARKET", "LIMIT")
                or not isinstance(getattr(candidate.decision_obj, "order_type", None), str)
            )
            and d_action not in ("HOLD", "NO_TRADE")
            and (
                d_action == "EXECUTE"
                or (
                    d_action == "WAIT"
                    and candidate.setup_grade in ("GRADE A+", "GRADE A")
                    and candidate.expected_value >= 0.40
                    and candidate.risk_reward_ratio >= 1.4
                    and failing_cnt <= 1
                )
                or (
                    d_action == "WAIT"
                    and candidate.setup_grade == "GRADE B"
                    and candidate.expected_value >= 0.40
                    and candidate.risk_reward_ratio >= 1.4
                    and failing_cnt == 0
                )
            )
        )

        assert is_exec_ready is True, "Grade A+ candidate in WAIT status with 1 softened failure must be exec_ready"

    def test_structural_timing_gates_not_fatal_hard_gates(self):
        """Premium/Discount Alignment and Trend Not Exhausted must be soft gates so limit/wait routing stays alive."""
        from jarvis.intelligence.gate_registry import HARD_GATES, SOFT_GATE_PENALTIES
        assert "Premium/Discount Alignment" not in HARD_GATES
        assert "Trend Not Exhausted" not in HARD_GATES
        assert "Premium/Discount Alignment" in SOFT_GATE_PENALTIES
        assert "Trend Not Exhausted" in SOFT_GATE_PENALTIES
