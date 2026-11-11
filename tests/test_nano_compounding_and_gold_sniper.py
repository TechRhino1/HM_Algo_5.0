"""
Unit and integration test suite for:
- Small Account ($100) Invariant Dollar Risk Sizing
- Forex 0.02 lot promotion on tight stops
- Gold (XAUUSD) 0.01 lot strict floor on sub-$150 accounts
- Gold 0.02 lot unlock at $150+ milestone tier
- Milestone Compounding & Risk Budgets
- Gold profile strategy weights & killzone windows
- Orchestrator splittable book protocol
"""
import unittest
from jarvis.risk.account_tier import (
    get_max_lot_cap,
    get_nano_risk_cap,
    get_milestone_risk_budget,
    AccountTier,
    get_account_tier,
)
from jarvis.risk.position_sizing import PositionSizer
from jarvis.intelligence.symbol_profile_config import get_symbol_profile_config


class TestNanoCompoundingAndGoldSniper(unittest.TestCase):

    def test_milestone_risk_budgets_progression(self):
        """Verifies deterministic progression of milestone risk budgets from $100 to $1,000+."""
        # Tier 1: $100 - $150
        fx_t1_usd, fx_t1_pct = get_milestone_risk_budget(100.0, "EURUSD")
        gold_t1_usd, gold_t1_pct = get_milestone_risk_budget(100.0, "XAUUSD")
        self.assertEqual(fx_t1_usd, 2.50)
        self.assertEqual(fx_t1_pct, 2.5)
        self.assertEqual(gold_t1_usd, 3.00)
        self.assertEqual(gold_t1_pct, 3.0)

        # Tier 2: $150 - $250
        fx_t2_usd, fx_t2_pct = get_milestone_risk_budget(180.0, "EURUSD")
        gold_t2_usd, gold_t2_pct = get_milestone_risk_budget(180.0, "XAUUSD")
        self.assertEqual(fx_t2_usd, 4.00)
        self.assertEqual(gold_t2_usd, 5.00)

        # Tier 3: $250 - $500
        fx_t3_usd, _ = get_milestone_risk_budget(300.0, "EURUSD")
        self.assertEqual(fx_t3_usd, 7.00)

        # Tier 4: $500 - $1,000
        fx_t4_usd, _ = get_milestone_risk_budget(750.0, "EURUSD")
        self.assertEqual(fx_t4_usd, 12.00)

    def test_max_lot_caps_by_tier_and_asset(self):
        """Verifies lot caps permit 0.02 on FX on $100 while keeping Gold strictly capped at 0.01."""
        # On $100 account:
        self.assertEqual(get_max_lot_cap(100.0, "EURUSD"), 0.02)
        self.assertEqual(get_max_lot_cap(100.0, "GBPUSD"), 0.02)
        self.assertEqual(get_max_lot_cap(100.0, "XAUUSD"), 0.01)  # Gold strictly capped!

        # On $160 account (Tier 2):
        self.assertEqual(get_max_lot_cap(160.0, "EURUSD"), 0.03)
        self.assertEqual(get_max_lot_cap(160.0, "XAUUSD"), 0.02)  # Gold unlocked for 0.02 split!

        # On $300 account (Tier 3):
        self.assertEqual(get_max_lot_cap(300.0, "EURUSD"), 0.05)
        self.assertEqual(get_max_lot_cap(300.0, "XAUUSD"), 0.05)

    def test_forex_tight_stop_sizes_to_002_lots_on_100_dollar_account(self):
        """Verifies EURUSD with an 8-pip stop sizes to 0.02 lots on $100 equity."""
        symbol_info = {
            "name": "EURUSD",
            "symbol": "EURUSD",
            "canonical": "EURUSD",
            "point": 0.00001,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "trade_contract_size": 100000.0,
            "trade_tick_value": 1.0,
            "trade_tick_size": 0.00001,
        }
        # Entry 1.0850, SL 1.0842 (8 pips distance = 0.0008)
        lots = PositionSizer.calculate_lot_size(
            account_balance=100.0,
            entry_price=1.0850,
            sl_price=1.0842,
            risk_pct=2.0,
            symbol_info=symbol_info,
            model_confidence=0.75,
            target_rr=2.5
        )
        self.assertEqual(lots, 0.02)
        # Dollar risk at 0.02 lots = 0.0008 * 100,000 * 0.02 = $1.60 (<= $2.50 milestone budget)
        dollar_risk = 0.0008 * 100000.0 * lots
        self.assertLessEqual(dollar_risk, 2.50)

    def test_gold_on_100_dollar_account_strictly_capped_at_001_lot(self):
        """Verifies Gold with tight stop is strictly capped at 0.01 lot on a $100 account."""
        symbol_info = {
            "name": "XAUUSD",
            "symbol": "XAUUSD",
            "canonical": "XAUUSD",
            "point": 0.01,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "trade_contract_size": 100.0,
            "trade_tick_value": 1.0,
            "trade_tick_size": 0.01,
        }
        # Entry 2650.00, SL 2648.00 ($2.00 distance)
        lots = PositionSizer.calculate_lot_size(
            account_balance=100.0,
            entry_price=2650.00,
            sl_price=2648.00,
            risk_pct=2.5,
            symbol_info=symbol_info,
            model_confidence=0.80,
            target_rr=3.0
        )
        self.assertEqual(lots, 0.01)
        dollar_risk = 2.00 * 100.0 * lots
        self.assertEqual(dollar_risk, 2.00)

    def test_gold_on_160_dollar_account_unlocks_002_lots_tier2(self):
        """Verifies Gold on a $160 account unlocks 0.02 lots for a tight $2.00 stop."""
        symbol_info = {
            "name": "XAUUSD",
            "symbol": "XAUUSD",
            "canonical": "XAUUSD",
            "point": 0.01,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "trade_contract_size": 100.0,
            "trade_tick_value": 1.0,
            "trade_tick_size": 0.01,
        }
        # Entry 2650.00, SL 2648.00 ($2.00 distance = $200 / lot risk)
        # On $160 account, Tier 2 budget is $5.00
        lots = PositionSizer.calculate_lot_size(
            account_balance=160.0,
            entry_price=2650.00,
            sl_price=2648.00,
            risk_pct=2.5,
            symbol_info=symbol_info,
            model_confidence=0.85,
            target_rr=3.0
        )
        self.assertEqual(lots, 0.02)
        dollar_risk = 2.00 * 100.0 * lots
        self.assertEqual(dollar_risk, 4.00)
        self.assertLessEqual(dollar_risk, 5.00)

    def test_gold_profile_configuration_and_killzones(self):
        """Verifies Gold profile has institutional settings and London/NY prime hours."""
        gold_prof = get_symbol_profile_config("XAUUSD")
        self.assertIsNotNone(gold_prof)
        self.assertEqual(gold_prof.strategy_weights.get("TREND_PULLBACK"), 4.0)
        self.assertEqual(gold_prof.strategy_weights.get("TREND_FOLLOWING"), 3.5)
        self.assertEqual(gold_prof.allowed_utc_hours, (7, 18))
        self.assertTrue(gold_prof.premium_window_exception)
        self.assertEqual(gold_prof.premium_out_of_window_min_ai_score, 85.0)
        self.assertEqual(gold_prof.fast_cash_r, 1.10)
        self.assertEqual(gold_prof.be_trigger_r, 1.20)

    def test_orchestrator_splittable_book_logic(self):
        """Verifies book splitting logic correctly identifies 0.02 lots as splittable."""
        from jarvis.application.orchestrator import JarvisOrchestrator
        from unittest.mock import MagicMock

        orch = JarvisOrchestrator.__new__(JarvisOrchestrator)
        orch.risk_engine = MagicMock()
        orch.risk_engine.max_risk_per_trade_pct = 0.5

        # EURUSD 8-pip stop on $100 account
        spec = MagicMock()
        spec.pip_size = 0.0001
        spec.pip_value_per_lot = 10.0

        # With 8 pips stop ($80/lot) and 2% risk budget ($2.00), ideal_lots = 2.00 / 80 = 0.025
        # 0.025 >= 2 * 0.01 = 0.02, so it is splittable (returns False for _is_unsplittable_book)
        is_unsplit = orch._is_unsplittable_book(
            entry_price=1.0850,
            stop_loss=1.0842,
            equity=100.0,
            spec=spec,
            volume_min=0.01
        )
        self.assertFalse(is_unsplit, "0.025 ideal lots must be recognized as splittable!")

        # With a 30-pip stop ($300/lot), ideal_lots = 2.00 / 300 = 0.0067 < 0.02 -> unsplittable
        is_unsplit_wide = orch._is_unsplittable_book(
            entry_price=1.0850,
            stop_loss=1.0820,
            equity=100.0,
            spec=spec,
            volume_min=0.01
        )
        self.assertTrue(is_unsplit_wide, "0.0067 ideal lots must be recognized as unsplittable.")


if __name__ == "__main__":
    unittest.main()
