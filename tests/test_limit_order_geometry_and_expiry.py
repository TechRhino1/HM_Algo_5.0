import unittest
from unittest.mock import MagicMock, patch
import time
from datetime import datetime, timezone

from jarvis.execution.execution_engine import ExecutionEngine
from jarvis.intelligence.dynamic_levels import DynamicRiskAndLevelsEngine
from jarvis.execution.mt5_client import MT5Client
from jarvis.execution.position_monitor import PositionMonitorEngine
from jarvis.data.schemas import (
    MarketContext,
    StructureContext,
    VolatilityContext,
    MomentumContext,
    LiquidityContext,
    SessionContext,
    RegimeOutput,
    MarketRegime
)


class MockDecision:
    def __init__(self, symbol, bias, entry_price, stop_loss, take_profit, order_type="LIMIT", style="SWING"):
        self.symbol = symbol
        self.bias = bias
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_type = order_type
        self.style = style
        self.execution_authorized = True
        self.calculated_risk_percent = 0.5
        self.first_target_price = None
        self.first_target_volume_pct = 0.5
        self.tp1_price = None
        self.tp2_price = take_profit
        self.tp3_price = None
        self.runner_trail_distance_atr = 1.5
        self.sl_distance = abs(entry_price - stop_loss)
        self.tp_distance = abs(entry_price - take_profit)
        self.risk_reward_ratio = 2.0


class TestLimitOrderGeometryAndExpiry(unittest.TestCase):
    def setUp(self):
        self.mock_mt5 = MagicMock()
        self.mock_mt5.mode = "live"
        self.mock_mt5.is_connected = True
        self.mock_mt5.magic_number = 888999
        self.mock_mt5.resolve_symbol_name.side_effect = lambda s: s
        self.mock_mt5.get_symbol_trading_spec.return_value = {
            "digits": 2,
            "trade_tick_size": 0.01,
            "point": 0.01,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "trade_stops_level": 5,
            "spread": 10
        }
        self.mock_mt5.get_live_spread_points.return_value = 10.0
        self.mock_mt5.get_broker_time.return_value = 1794410000.0
        self.mock_mt5.get_pending_orders.return_value = []

        self.mock_risk = MagicMock()
        self.mock_risk.calculate_position_size.return_value = 0.01
        self.mock_risk.validate_order.return_value = {"allowed": True, "reason": "OK"}

        self.mock_state = MagicMock()
        self.mock_state.is_safe_mode = False
        self.mock_state.safe_mode = False
        self.mock_state.positions = []
        self.mock_state.account.equity = 10000.0
        self.mock_state.account.balance = 10000.0
        self.mock_state.account.currency = "USD"
        self.mock_state.is_live_allowed.return_value = True

        self.engine = ExecutionEngine(
            mt5_client=self.mock_mt5,
            state_manager=self.mock_state
        )

    def test_sell_limit_distance_clamped_when_too_far(self):
        """Verify that a SELL_LIMIT placed far above market is clamped to max_limit_dist."""
        ctx = MarketContext(
            symbol="ETHUSD",
            current_price=2050.0,
            ask=2050.5,
            bid=2049.5,
            structure=StructureContext(bias="BEARISH"),
            volatility=VolatilityContext(atr=50.0),
            momentum=MomentumContext(),
            timestamp=datetime.now(timezone.utc),
            liquidity=LiquidityContext(buy_side_liquidity=2200.0, sell_side_liquidity=1700.0),
            session=SessionContext()
        )

        # Intended entry 90 pts above market (2140 vs 2050), TP is 1700 (tp_dist = 440 pts; 90/440 = 20% traversed)
        decision = MockDecision(
            symbol="ETHUSD",
            bias="SELL",
            entry_price=2140.0,
            stop_loss=2190.0,
            take_profit=1700.0,
            order_type="LIMIT",
            style="SWING"
        )
        decision.context = ctx

        res = self.engine.execute_decision(decision, 0.01)
        self.mock_mt5.place_pending_order.assert_called_once()
        call_kwargs = self.mock_mt5.place_pending_order.call_args.kwargs
        # Max limit distance for swing is min(1.35 * 50, 2050 * 0.03) = min(67.5, 61.5) = 61.5 pts
        # So clamped entry must be ~ 2050.5 + 61.5 = 2112.0, NOT 2140.0!
        placed_price = call_kwargs["price"]
        self.assertLess(placed_price, 2125.0, f"Entry {placed_price} should have been clamped near market")
        self.assertGreater(placed_price, 2050.0)
        # SL and TP must maintain the structural distances
        self.assertEqual(call_kwargs["order_type"], "SELL_LIMIT")
        self.assertGreater(call_kwargs["sl_price"], placed_price)
        self.assertLess(call_kwargs["tp_price"], placed_price)
        self.assertEqual(call_kwargs["expiration_sec"], 2400)

    def test_buy_limit_distance_clamped_when_too_far(self):
        """Verify that a BUY_LIMIT placed far below market is clamped to max_limit_dist."""
        ctx = MarketContext(
            symbol="EURUSD",
            current_price=1.1100,
            ask=1.1101,
            bid=1.1099,
            structure=StructureContext(bias="BULLISH"),
            volatility=VolatilityContext(atr=0.0030),
            momentum=MomentumContext(),
            timestamp=datetime.now(timezone.utc),
            liquidity=LiquidityContext(),
            session=SessionContext()
        )
        self.mock_mt5.get_symbol_trading_spec.return_value = {
            "digits": 5,
            "point": 0.00001,
            "trade_tick_size": 0.00001,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "trade_stops_level": 5,
            "spread": 10
        }

        # Entry 60 pips below market (1.1040 vs 1.1100), TP is 1.1300 (tp_dist = 260 pips; 60/260 = 23% traversed)
        decision = MockDecision(
            symbol="EURUSD",
            bias="BUY",
            entry_price=1.1040,
            stop_loss=1.1010,
            take_profit=1.1300,
            order_type="LIMIT",
            style="DAY_TRADING"
        )
        decision.context = ctx

        res = self.engine.execute_decision(decision, 0.01)
        self.mock_mt5.place_pending_order.assert_called_once()
        call_kwargs = self.mock_mt5.place_pending_order.call_args.kwargs
        placed_price = call_kwargs["price"]
        # Max limit distance for Day is min(0.85 * 0.0030 = 0.00255, 1.1100 * 0.015 = 0.0166) = 0.00255 (25.5 pips)
        # Entry should be clamped near 1.1099 - 0.00255 + spread = ~1.10755, not 1.1040!
        self.assertGreater(placed_price, 1.1065, f"Entry {placed_price} should have been clamped near market")
        self.assertLess(placed_price, 1.1099)
        self.assertEqual(call_kwargs["order_type"], "BUY_LIMIT")
        self.assertEqual(call_kwargs["expiration_sec"], 1200)

    def test_missed_the_boat_blocked_in_execution_engine(self):
        """Verify that when market has already covered >= 40% towards TP, resting limit is blocked."""
        self.mock_state.account.equity = 50000.0
        self.mock_state.account.balance = 50000.0
        ctx = MarketContext(
            symbol="GOLD",
            current_price=4039.0,
            ask=4039.5,
            bid=4039.0,
            structure=StructureContext(bias="BEARISH"),
            volatility=VolatilityContext(atr=8.0),
            momentum=MomentumContext(),
            timestamp=datetime.now(timezone.utc),
            liquidity=LiquidityContext(),
            session=SessionContext()
        )

        # Origin entry was 4052, TP is 4033 (tp_dist = 19 pts). Current ask is 4039.5 (12.5 pts traversed = 65%!)
        decision = MockDecision(
            symbol="GOLD",
            bias="SELL",
            entry_price=4052.0,
            stop_loss=4060.0,
            take_profit=4033.0,
            order_type="LIMIT",
            style="DAY_TRADING"
        )
        decision.context = ctx

        res = self.engine.execute_decision(decision, 0.01)
        self.assertEqual(res.get("status"), "BLOCKED")
        self.assertEqual(res.get("reason"), "LIMIT_SETUP_MISSED_BOAT_NEAR_TP")
        self.mock_mt5.place_pending_order.assert_not_called()

    def test_crypto_short_tp_bounded_in_dynamic_levels(self):
        """Verify that dynamic levels for ETH short never returns extreme targets like $52."""
        levels_calc = DynamicRiskAndLevelsEngine()
        ctx = MarketContext(
            symbol="ETHUSD",
            current_price=2374.0,
            ask=2374.5,
            bid=2373.5,
            structure=StructureContext(bias="BEARISH", demand_zone=(2100.0, 2150.0), supply_zone=(2550.0, 2600.0)),
            volatility=VolatilityContext(atr=150.0),
            momentum=MomentumContext(trend_score=-80.0, adx=35.0),
            timestamp=datetime.now(timezone.utc),
            liquidity=LiquidityContext(),
            session=SessionContext()
        )
        regime = RegimeOutput(
            primary_regime=MarketRegime.TREND_BEAR,
            confidence=0.85,
            probabilities={}
        )

        levels = levels_calc.calculate_levels(
            context=ctx,
            regime=regime,
            tentative_bias="SELL",
            account_balance=10000.0,
            risk_per_trade_pct=0.5,
            trade_style="SWING"
        )

        tp_price = levels["tp_price"]
        entry_price = levels["entry_price"]
        # Max crypto drop for swing is 20%. Minimum floor is entry * 0.80 = 2374 * 0.80 = ~1899.20
        self.assertGreaterEqual(tp_price, entry_price * 0.79, f"TP {tp_price} should never be lower than 80% of entry {entry_price}")
        self.assertLess(tp_price, entry_price)

    def test_position_monitor_server_time_age_calculation(self):
        """Verify position monitor calculates age accurately using broker server time."""
        mock_mt5 = MagicMock()
        mock_mt5.get_broker_time.return_value = 1794410500.0  # Server time
        # Order setup time was 1794410000.0 (500s ago in server time)
        mock_mt5.get_pending_orders.return_value = [{
            "ticket": 123456,
            "symbol": "EURUSD",
            "type": "SELL_LIMIT",
            "price_open": 1.1150,
            "sl": 1.1180,
            "tp": 1.1050,
            "comment": "HMA2_DAY_TREN_LIVE",
            "time_setup": 1794410000.0
        }]
        state_mgr = MagicMock()
        state_mgr.positions = []

        monitor = PositionMonitorEngine(
            mt5_client=mock_mt5,
            data_feed=MagicMock(),
            context_engine=MagicMock(),
            state_manager=state_mgr
        )

        with patch("time.time", return_value=1794403300.0):  # Local time with 7200s skew
            monitor._manage_pending_orders_tick()

        # Since age is 500s and allowed_age is 1200s for DAY, order should NOT be cancelled yet
        mock_mt5.cancel_pending_order.assert_not_called()

        # Advance server time past 1200s
        mock_mt5.get_broker_time.return_value = 1794411250.0  # 1250s after setup
        with patch("time.time", return_value=1794404050.0):
            monitor._manage_pending_orders_tick()

        # Now age (1250s) > 1200s, so it must be cancelled!
        mock_mt5.cancel_pending_order.assert_called_once_with(123456)


if __name__ == "__main__":
    unittest.main()
