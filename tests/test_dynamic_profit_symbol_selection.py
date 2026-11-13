"""
Tests for Dynamic Market Conduction & High-Profit Priority Symbol Selection in HM Algo 2.0.
Verifies:
1. Dynamic Market Conduction scoring across ADX, volatility state, spread friction, OTE zone, and prime session.
2. Dynamic Profit Potential scoring across EV_R, EV_$, R:R asymmetry, and ATR expansion.
3. Super-Priority Tier 0 promotion of elite profit opportunities across FX, Crypto, and Metals.
4. Clean fallback to baseline profile order when market context is absent (100% backward compatibility).
5. Radar sort key ordering alignment with dynamic priority scores.
"""
import unittest
from datetime import datetime, timezone

from jarvis.data.schemas import (
    DecisionObject,
    MarketContext,
    MarketRegime,
    MomentumContext,
    RegimeOutput,
    SessionContext,
    StructureContext,
    TradeQualityGateResult,
    VolatilityContext,
)
from jarvis.intelligence.opportunity_arbiter import (
    CandidateOpportunity,
    UniversalOpportunityArbiter,
)
from jarvis.application.radar_sort import radar_sort_key


class TestDynamicProfitSymbolSelection(unittest.TestCase):

    def setUp(self):
        self.arbiter = UniversalOpportunityArbiter(ml_predictor=None, bandit=None)

    def _make_context(
        self,
        symbol="EURUSD",
        current_price=1.0850,
        adx=32.0,
        trend_score=45,
        vol_state="EXPANSION",
        spread_pips=0.8,
        zone="DISCOUNT",
        is_prime=True,
        atr=0.0060,
    ) -> MarketContext:
        return MarketContext(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            current_price=current_price,
            bid=current_price - (spread_pips * 0.0001 / 2),
            ask=current_price + (spread_pips * 0.0001 / 2),
            structure=StructureContext(
                bias="BULLISH",
                discount_premium_zone=zone,
            ),
            liquidity=None,
            volatility=VolatilityContext(
                atr=atr,
                current_spread_pips=spread_pips,
                state=vol_state,
            ),
            momentum=MomentumContext(
                adx=adx,
                trend_score=trend_score,
            ),
            session=SessionContext(
                current_session="LONDON",
                is_prime_session=is_prime,
            ),
        )

    def _make_decision(
        self,
        symbol="EURUSD",
        bias="BUY",
        entry_price=1.0850,
        stop_loss=1.0820,
        take_profit=1.0940,
        expected_value=1.5,
        ev_dollars=15.0,
        rr_ratio=3.0,
        ai_score=85.0,
    ) -> DecisionObject:
        return DecisionObject(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            regime=RegimeOutput(
                primary_regime=MarketRegime.TREND_BULL,
                probabilities={"TREND_BULL": 0.85},
                confidence=0.85,
            ),
            bias=bias,
            probabilities={"buy": 0.75, "sell": 0.25},
            strategy="TREND_PULLBACK",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr_ratio,
            calculated_risk_percent=1.0,
            expected_value=expected_value,
            expected_value_dollars=ev_dollars,
            model_confidence=0.75,
            adversarial_penalty=0.0,
            invalidation_levels=[],
            bull_case=["Clean trend", "Discount OTE"],
            bear_case=[],
            risk_factors=[],
            quality_gate=TradeQualityGateResult(passed=True, checks={}),
            decision="EXECUTE",
            ai_score=ai_score,
        )

    def test_market_conduction_calculation_prime_conditions(self):
        """Pristine trending market with expansion, tight spread, and discount entry must yield high conduction."""
        ctx = self._make_context(symbol="EURUSD", adx=35.0, trend_score=50, vol_state="EXPANSION", spread_pips=0.6, zone="DISCOUNT", is_prime=True)
        dec = self._make_decision(symbol="EURUSD", bias="BUY", entry_price=1.0850, stop_loss=1.0810)
        cond = self.arbiter.calculate_market_conduction(symbol="EURUSD", context=ctx, decision_obj=dec, style="DAY_TRADING")
        self.assertGreaterEqual(cond, 1.25)

    def test_market_conduction_calculation_sluggish_conditions(self):
        """Choppy market with dead ADX, compression, and high friction must penalize conduction."""
        ctx = self._make_context(symbol="EURUSD", adx=14.0, trend_score=-10, vol_state="COMPRESSION", spread_pips=3.0, zone="PREMIUM", is_prime=False)
        dec = self._make_decision(symbol="EURUSD", bias="BUY", entry_price=1.0850, stop_loss=1.0840)
        cond = self.arbiter.calculate_market_conduction(symbol="EURUSD", context=ctx, decision_obj=dec, style="DAY_TRADING")
        self.assertLess(cond, 0.90)

    def test_market_conduction_defaults_to_one_when_context_none(self):
        """When context is absent, conduction defaults cleanly to 1.0."""
        cond = self.arbiter.calculate_market_conduction(symbol="EURUSD", context=None, decision_obj=None)
        self.assertEqual(cond, 1.0)

    def test_profit_potential_scaling(self):
        """High EV and asymmetric RR must produce elevated profit potential score."""
        ctx = self._make_context(symbol="BTCUSD", current_price=65000.0, atr=2000.0)
        dec = self._make_decision(symbol="BTCUSD", expected_value=2.2, ev_dollars=50.0, rr_ratio=3.5)
        pot = self.arbiter.calculate_profit_potential(expected_value=2.2, rr_ratio=3.5, win_prob=70.0, symbol="BTCUSD", context=ctx, decision_obj=dec)
        self.assertGreaterEqual(pot, 1.40)

    def test_elite_fx_dynamically_beats_lower_conduction_gold(self):
        """An elite EURUSD opportunity with prime market conduction and high EV must beat a choppy Gold setup."""
        # Setup 1: Choppy Gold setup (off-hours, sluggish ADX, low EV)
        gold_ctx = self._make_context(symbol="XAUUSD", current_price=2600.0, adx=15.0, trend_score=10, vol_state="COMPRESSION", spread_pips=3.5, zone="PREMIUM", is_prime=False, atr=15.0)
        gold_dec = self._make_decision(symbol="XAUUSD", bias="BUY", entry_price=2600.0, stop_loss=2595.0, take_profit=2607.0, expected_value=0.4, ev_dollars=4.0, rr_ratio=1.4)
        gold_cand = self.arbiter.evaluate_opportunity(gold_dec, gold_ctx, trade_style="SWING")

        # Setup 2: Elite EURUSD setup (London session, strong ADX, deep discount, high EV)
        fx_ctx = self._make_context(symbol="EURUSD", current_price=1.0850, adx=36.0, trend_score=60, vol_state="EXPANSION", spread_pips=0.7, zone="DISCOUNT", is_prime=True, atr=0.0070)
        fx_dec = self._make_decision(symbol="EURUSD", bias="BUY", entry_price=1.0850, stop_loss=1.0815, take_profit=1.0955, expected_value=1.8, ev_dollars=25.0, rr_ratio=3.0)
        fx_cand = self.arbiter.evaluate_opportunity(fx_dec, fx_ctx, trade_style="DAY_TRADING")

        # Ensure both are actionable Grade A/A+
        self.assertTrue(fx_cand.is_actionable)
        self.assertIn(fx_cand.setup_grade, ("GRADE A+", "GRADE A"))

        # Verify arbitration elevates EURUSD to #1
        best, ranked = self.arbiter.rank_and_select_best([gold_cand, fx_cand])
        self.assertIsNotNone(best)
        self.assertEqual(best.symbol, "EURUSD")
        self.assertEqual(ranked[0].symbol, "EURUSD")
        self.assertGreater(fx_cand.dynamic_priority_score, gold_cand.dynamic_priority_score)

    def test_elite_crypto_dynamically_beats_lower_conduction_gold(self):
        """An elite BTCUSD breakout opportunity in prime conduction must beat a choppy Gold setup."""
        gold_ctx = self._make_context(symbol="XAUUSD", current_price=2600.0, adx=16.0, trend_score=5, vol_state="COMPRESSION", spread_pips=3.5, zone="PREMIUM", is_prime=False)
        gold_dec = self._make_decision(symbol="XAUUSD", bias="BUY", entry_price=2600.0, stop_loss=2595.0, take_profit=2607.0, expected_value=0.4, ev_dollars=4.0, rr_ratio=1.4)
        gold_cand = self.arbiter.evaluate_opportunity(gold_dec, gold_ctx, trade_style="SWING")

        btc_ctx = self._make_context(symbol="BTCUSD", current_price=65000.0, adx=38.0, trend_score=65, vol_state="EXPANSION", spread_pips=50.0, zone="DISCOUNT", is_prime=True, atr=1800.0)
        btc_dec = self._make_decision(symbol="BTCUSD", bias="BUY", entry_price=65000.0, stop_loss=64000.0, take_profit=68500.0, expected_value=2.1, ev_dollars=60.0, rr_ratio=3.5)
        btc_cand = self.arbiter.evaluate_opportunity(btc_dec, btc_ctx, trade_style="DAY_TRADING")

        best, ranked = self.arbiter.rank_and_select_best([gold_cand, btc_cand])
        self.assertEqual(best.symbol, "BTCUSD")
        self.assertEqual(ranked[0].symbol, "BTCUSD")

    def test_elite_gold_still_beats_fx_when_gold_has_superior_conduction(self):
        """When Gold itself has an elite setup with higher dynamic priority, Gold takes #1."""
        fx_ctx = self._make_context(symbol="EURUSD", current_price=1.0850, adx=26.0, trend_score=35, vol_state="NORMAL", spread_pips=1.0, zone="DISCOUNT", is_prime=True)
        fx_dec = self._make_decision(symbol="EURUSD", bias="BUY", entry_price=1.0850, stop_loss=1.0820, take_profit=1.0920, expected_value=1.1, ev_dollars=12.0, rr_ratio=2.3)
        fx_cand = self.arbiter.evaluate_opportunity(fx_dec, fx_ctx, trade_style="DAY_TRADING")

        gold_ctx = self._make_context(symbol="XAUUSD", current_price=2600.0, adx=42.0, trend_score=75, vol_state="EXPANSION", spread_pips=1.8, zone="DISCOUNT", is_prime=True, atr=25.0)
        gold_dec = self._make_decision(symbol="XAUUSD", bias="BUY", entry_price=2600.0, stop_loss=2585.0, take_profit=2645.0, expected_value=2.4, ev_dollars=70.0, rr_ratio=3.0)
        gold_cand = self.arbiter.evaluate_opportunity(gold_dec, gold_ctx, trade_style="SWING")

        best, ranked = self.arbiter.rank_and_select_best([fx_cand, gold_cand])
        self.assertEqual(best.symbol, "XAUUSD")
        self.assertEqual(ranked[0].symbol, "XAUUSD")

    def test_legacy_synthetic_fallback_unbroken(self):
        """When candidates are constructed synthetically without context, baseline profile order is 100% preserved."""
        fx = CandidateOpportunity(
            symbol="EURUSD", trade_style="SWING", timeframe="D1/H4/H1", strategy="TREND_PULLBACK",
            bias="BUY", entry_price=1.0850, stop_loss=1.0820, take_profit=1.0910, risk_reward_ratio=2.0,
            expected_value=2.0, win_prob=90.0, ml_prob=0.8, confluence_score=90.0, confluence_factors=["test"],
            adversarial_penalty=0.0, risk_factors=[], regime="TREND_BULL", regime_multiplier=1.0,
            utility_score=2.50, setup_grade="GRADE A+", is_actionable=True
        )
        gold = CandidateOpportunity(
            symbol="XAUUSD", trade_style="SWING", timeframe="D1/H4/H1", strategy="TREND_PULLBACK",
            bias="BUY", entry_price=2600.0, stop_loss=2590.0, take_profit=2620.0, risk_reward_ratio=2.0,
            expected_value=2.0, win_prob=90.0, ml_prob=0.8, confluence_score=90.0, confluence_factors=["test"],
            adversarial_penalty=0.0, risk_factors=[], regime="TREND_BULL", regime_multiplier=1.0,
            utility_score=1.05, setup_grade="GRADE A+", is_actionable=True
        )
        best, ranked = self.arbiter.rank_and_select_best([fx, gold])
        self.assertEqual(best.symbol, "XAUUSD")  # Baseline fallback preserves Gold
        self.assertEqual(ranked[0].symbol, "XAUUSD")

    def test_radar_sort_key_orders_by_dynamic_priority(self):
        """Radar display sort key orders higher dynamic priority score first."""
        item_high = {"action": "BUY READY", "dynamic_priority_score": 3.5, "utility_score": 1.2, "win_prob": 70, "ev": 1.5}
        item_low = {"action": "BUY READY", "dynamic_priority_score": 1.1, "utility_score": 1.5, "win_prob": 70, "ev": 1.5}
        key_high = radar_sort_key(item_high)
        key_low = radar_sort_key(item_low)
        self.assertGreater(key_high, key_low)


if __name__ == "__main__":
    unittest.main()
