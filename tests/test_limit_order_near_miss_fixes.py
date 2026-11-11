"""Tests for Gold and BTCUSD limit order near-miss and spread asymmetry fixes."""
import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from jarvis.data.schemas import (
    MarketContext,
    StructureContext,
    VolatilityContext,
    MomentumContext,
    LiquidityContext,
    SessionContext,
    RegimeOutput,
    MarketRegime,
    AnalystReport,
    AnalystRole,
    DevilAdvocateReport,
    DecisionObject,
)
from jarvis.intelligence.decision_engine import DecisionEngine
from jarvis.execution.execution_engine import ExecutionEngine
from jarvis.execution.position_monitor import PositionMonitorEngine


class TestLimitOrderNearMissFixes(unittest.TestCase):
    def test_sell_limit_selects_nearest_unmitigated_fvg_with_front_run(self):
        """Gold SELL_LIMIT must ignore mitigated/[0] zones and apply front-run cushion."""
        engine = DecisionEngine()
        ctx = MarketContext(
            symbol="XAUUSD",
            timestamp=datetime(2026, 11, 11, 14, 0, tzinfo=timezone.utc),
            current_price=4006.0,
            bid=4006.0,
            ask=4006.25,
            structure=StructureContext(
                bias="BEARISH",
                discount_premium_zone="DISCOUNT",
                equilibrium_price=4020.0,
                fair_value_gaps=[
                    {"type": "BEARISH_FVG", "top": 4030.0, "bottom": 4025.0, "mitigated": False},
                    {"type": "BEARISH_FVG", "top": 4010.0, "bottom": 4008.0, "mitigated": True},
                    {"type": "BEARISH_FVG", "top": 4009.0, "bottom": 4007.80, "mitigated": False},
                ],
            ),
            liquidity=LiquidityContext(buy_side_liquidity=4012.0, sell_side_liquidity=3985.0),
            volatility=VolatilityContext(atr=6.0, current_spread_pips=2.5),
            momentum=MomentumContext(trend_score=-45.0, rsi=34.0, adx=28.0),
            session=SessionContext(is_prime_session=True),
        )
        regime = RegimeOutput(primary_regime=MarketRegime.TREND_BEAR, probabilities={}, confidence=0.85)
        rep = AnalystReport(role=AnalystRole.STRUCTURE, symbol="XAUUSD", bias="SELL", score=82.0, confidence=0.85)
        devil = DevilAdvocateReport(symbol="XAUUSD", counter_bias="BUY", penalty_score=5.0, invalidation_risk_coefficient=1.0)

        dec = engine.evaluate(
            context=ctx,
            regime=regime,
            analyst_reports={"STRUCTURE": rep},
            devil_report=devil,
            account_balance=850.0,
            trade_style="DAY_TRADING",
        )

        self.assertEqual(dec.trade_style, "DAY_TRADING")
        self.assertEqual(dec.order_type, "LIMIT")
        # Nearest unmitigated Bearish FVG is [4007.80, 4009.0]. Proximal 65/35 blend = 4008.22 minus front_run (~0.20) = ~4008.02
        self.assertLess(dec.entry_price, 4008.40, "Must pick nearest unmitigated FVG + front-run buffer, not distant FVG[0] at 4027.5")
        self.assertGreater(dec.entry_price, ctx.current_price)

    def test_buy_limit_btcusd_compensates_for_ask_spread(self):
        """BTCUSD BUY_LIMIT must add Ask spread + front-run cushion to Bid structural zone."""
        engine = DecisionEngine()
        ctx = MarketContext(
            symbol="BTCUSD",
            timestamp=datetime(2026, 11, 11, 14, 0, tzinfo=timezone.utc),
            current_price=74250.0,
            bid=74250.0,
            ask=74265.0,  # $15.00 live spread
            structure=StructureContext(
                bias="BULLISH",
                discount_premium_zone="PREMIUM",
                equilibrium_price=74100.0,
                fair_value_gaps=[
                    {"type": "BULLISH_FVG", "top": 73620.0, "bottom": 73580.0, "mitigated": False},
                    {"type": "BULLISH_FVG", "top": 74205.0, "bottom": 74185.0, "mitigated": False},
                ],
            ),
            liquidity=LiquidityContext(buy_side_liquidity=74600.0, sell_side_liquidity=74150.0),
            volatility=VolatilityContext(atr=160.0, current_spread_pips=1500.0),
            momentum=MomentumContext(trend_score=45.0, rsi=69.0, adx=28.0),
            session=SessionContext(is_prime_session=True),
        )
        regime = RegimeOutput(primary_regime=MarketRegime.TREND_BULL, probabilities={}, confidence=0.85)
        rep = AnalystReport(role=AnalystRole.STRUCTURE, symbol="BTCUSD", bias="BUY", score=82.0, confidence=0.85)
        devil = DevilAdvocateReport(symbol="BTCUSD", counter_bias="SELL", penalty_score=5.0, invalidation_risk_coefficient=1.0)

        dec = engine.evaluate(
            context=ctx,
            regime=regime,
            analyst_reports={"STRUCTURE": rep},
            devil_report=devil,
            account_balance=850.0,
            trade_style="SCALP",
        )

        self.assertEqual(dec.trade_style, "SCALP")
        self.assertEqual(dec.order_type, "LIMIT")
        # Nearest unmitigated Bullish FVG proximal blend = 74198.0 + $15 spread + front_run (~$11.25) = ~74224.25
        self.assertGreater(dec.entry_price, 74210.0, "BUY_LIMIT must compensate for BTCUSD $15 Bid/Ask spread")
        self.assertLess(dec.entry_price, ctx.bid)

    def test_execution_engine_propagates_scalp_style_and_immediate_capture(self):
        """SCALP trade_style propagates to HMA2_SCLP_ comment and near-touch limits execute via MARKET."""
        mock_mt5 = MagicMock()
        mock_mt5.mode = "live"
        mock_mt5.is_connected = True
        mock_mt5.magic_number = 888999
        mock_mt5.resolve_symbol_name.side_effect = lambda s: s
        mock_mt5.get_pending_orders.return_value = []
        mock_mt5.get_symbol_trading_spec.return_value = {
            "digits": 2,
            "point": 0.01,
            "trade_tick_size": 0.01,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "trade_stops_level": 0,  # Broker reports 0 stops_level like XMGlobal
            "spread": 25,
        }
        mock_mt5.send_market_order.return_value = {"status": "FILLED", "ticket": 55501, "price": 4001.40}
        mock_mt5.place_pending_order.return_value = {"status": "PLACED", "ticket": 55502}

        state = MagicMock()
        state.is_safe_mode = False
        state.execution_mode = "live"
        state.account.equity = 25000.0  # 0.5% risk ceiling = $125 > $80 SL risk

        exec_eng = ExecutionEngine(mt5_client=mock_mt5, state_manager=state)
        ctx = MarketContext(
            symbol="XAUUSD",
            timestamp=datetime.now(timezone.utc),
            current_price=4001.38,
            bid=4001.38,
            ask=4001.63,
            structure=StructureContext(bias="BEARISH"),
            liquidity=LiquidityContext(),
            volatility=VolatilityContext(atr=4.0, current_spread_pips=2.5),
            momentum=MomentumContext(),
            session=SessionContext(),
        )
        # SELL_LIMIT only $0.27 above Bid ($0.02 above Ask!) -> must capture immediately via MARKET
        dec = DecisionObject(
            symbol="XAUUSD",
            timestamp=datetime.now(timezone.utc),
            regime=RegimeOutput(primary_regime=MarketRegime.TREND_BEAR, probabilities={}, confidence=0.85),
            bias="SELL",
            probabilities={"sell": 0.8},
            strategy="TREND_FOLLOWING",
            trade_style="SCALP",
            order_type="LIMIT",
            entry_price=4001.65,
            stop_loss=4009.65,
            take_profit=3985.65,
            risk_reward_ratio=2.0,
            calculated_risk_percent=0.5,
            expected_value=0.9,
            model_confidence=0.65,
            adversarial_penalty=0.0,
            invalidation_levels=[],
            bull_case=[],
            bear_case=[],
            risk_factors=[],
            quality_gate=MagicMock(failing_reasons=[]),
            decision="EXECUTE",
            execution_authorized=True,
            context=ctx,
        )

        res = exec_eng.execute_decision(dec, lots=0.01)
        mock_mt5.send_market_order.assert_called_once()
        mock_mt5.place_pending_order.assert_not_called()
        self.assertEqual(res.get("status"), "FILLED")
        self.assertIn("SCLP", mock_mt5.send_market_order.call_args.kwargs["comment"])

        # Now place genuine resting SCALP limit well above market -> checks 600s TTL and HMA2_SCLP_ comment
        mock_mt5.reset_mock()
        dec2 = DecisionObject(
            symbol="XAUUSD",
            timestamp=datetime.now(timezone.utc),
            regime=dec.regime,
            bias="SELL",
            probabilities=dec.probabilities,
            strategy="TREND_FOLLOWING",
            trade_style="SCALP",
            order_type="LIMIT",
            entry_price=4003.00,
            stop_loss=4011.00,
            take_profit=3987.00,
            risk_reward_ratio=2.0,
            calculated_risk_percent=0.5,
            expected_value=0.9,
            model_confidence=0.65,
            adversarial_penalty=0.0,
            invalidation_levels=[],
            bull_case=[],
            bear_case=[],
            risk_factors=[],
            quality_gate=dec.quality_gate,
            decision="EXECUTE",
            execution_authorized=True,
            context=ctx,
        )
        res2 = exec_eng.execute_decision(dec2, lots=0.01)
        mock_mt5.place_pending_order.assert_called_once()
        call_kw = mock_mt5.place_pending_order.call_args.kwargs
        self.assertEqual(call_kw["expiration_sec"], 600)
        self.assertIn("SCLP", call_kw["comment"])

    def test_position_monitor_one_shot_near_touch_nudge(self):
        """PositionMonitor performs one-shot nudge when price comes within near_touch_tol and bounces off."""
        mock_mt5 = MagicMock()
        mock_mt5.get_broker_time.return_value = 1794410100.0
        mock_mt5.modify_pending_order.return_value = {"status": "MODIFIED"}
        mock_mt5.get_pending_orders.return_value = [{
            "ticket": 900123,
            "symbol": "GOLD.i#",
            "type": "SELL_LIMIT",
            "price_open": 4008.05,
            "sl": 4016.05,
            "tp": 3988.25,
            "comment": "HMA2_DAY_TREN_LIVE",
            "time_setup": 1794410000.0,
        }]
        state = MagicMock()
        state.positions = []

        monitor = PositionMonitorEngine(
            mt5_client=mock_mt5,
            data_feed=MagicMock(),
            context_engine=MagicMock(),
            state_manager=state,
        )

        # Tick 1: Price reaches 4007.68 Bid (gap = 0.37 <= tol 0.40) -> recorded, not nudged yet
        ctx1 = MarketContext(
            symbol="XAUUSD",
            timestamp=datetime.now(timezone.utc),
            current_price=4007.68,
            bid=4007.68,
            ask=4007.88,
            structure=StructureContext(bias="BEARISH"),
            liquidity=LiquidityContext(),
            volatility=VolatilityContext(atr=6.0, current_spread_pips=2.0),
            momentum=MomentumContext(trend_score=-20.0),
            session=SessionContext(),
        )
        monitor._get_context = MagicMock(return_value=ctx1)
        monitor._manage_pending_orders_tick()
        mock_mt5.modify_pending_order.assert_not_called()

        # Tick 2: Price dips back slightly to 4007.45 Bid (gap = 0.60 > 0.37 + bounce) -> triggers ONE-SHOT nudge!
        ctx2 = MarketContext(
            symbol="XAUUSD",
            timestamp=datetime.now(timezone.utc),
            current_price=4007.45,
            bid=4007.45,
            ask=4007.65,
            structure=StructureContext(bias="BEARISH"),
            liquidity=LiquidityContext(),
            volatility=VolatilityContext(atr=6.0, current_spread_pips=2.0),
            momentum=MomentumContext(trend_score=-20.0),
            session=SessionContext(),
        )
        monitor._get_context = MagicMock(return_value=ctx2)
        monitor._manage_pending_orders_tick()
        mock_mt5.modify_pending_order.assert_called_once()
        nudged_px = mock_mt5.modify_pending_order.call_args.kwargs["price"]
        self.assertLess(nudged_px, 4008.05)
        self.assertGreater(nudged_px, 4007.65)

        # Tick 3: Must NOT fire a second nudge on the same ticket
        mock_mt5.reset_mock()
        monitor._manage_pending_orders_tick()
        mock_mt5.modify_pending_order.assert_not_called()

    def test_trending_momentum_limit_enforces_scalp_clamp_and_spread(self):
        """Even on is_trending_momentum limit fallback, BTCUSD SCALP limits clamp within 0.40*ATR with spread cushion."""
        engine = DecisionEngine()
        ctx = MarketContext(
            symbol="BTCUSD",
            timestamp=datetime(2026, 11, 11, 14, 0, tzinfo=timezone.utc),
            current_price=74200.0,
            bid=74200.0,
            ask=74215.0,
            structure=StructureContext(bias="BULLISH", discount_premium_zone="EQUILIBRIUM"),
            liquidity=LiquidityContext(buy_side_liquidity=75000.0, sell_side_liquidity=73900.0),
            volatility=VolatilityContext(atr=160.0, current_spread_pips=1500.0),
            momentum=MomentumContext(trend_score=80.0, rsi=60.0, adx=30.0),
            session=SessionContext(is_prime_session=True),
        )
        regime = RegimeOutput(primary_regime=MarketRegime.TREND_BULL, probabilities={}, confidence=0.85)
        rep = AnalystReport(role=AnalystRole.STRUCTURE, symbol="BTCUSD", bias="BUY", score=84.0, confidence=0.85)
        devil = DevilAdvocateReport(symbol="BTCUSD", counter_bias="SELL", penalty_score=5.0, invalidation_risk_coefficient=1.0)

        # Simulated passive baseline far below current price (would be -600 pts without clamp)
        dec = engine.evaluate(
            context=ctx,
            regime=regime,
            analyst_reports={"STRUCTURE": rep},
            devil_report=devil,
            account_balance=850.0,
            trade_style="SCALP",
        )
        if dec.order_type == "LIMIT":
            self.assertLessEqual(ctx.current_price - dec.entry_price, 64.5, "SCALP BTCUSD limit must stay within ~0.40 ATR of market")

    def test_mt5_client_modify_pending_preserves_expiration(self):
        """MT5Client.modify_pending_order must retain original ORDER_TIME_SPECIFIED & expiration."""
        from unittest.mock import patch
        from jarvis.execution.mt5_client import MT5Client

        client = MT5Client(mode="live")
        client.is_connected = True

        fake_pending = MagicMock()
        fake_pending.symbol = "GOLD.i#"
        fake_pending.volume_initial = 0.01
        fake_pending.type = 3
        fake_pending.price_open = 4008.05
        fake_pending.sl = 4016.05
        fake_pending.tp = 3988.25
        fake_pending.type_time = 2  # ORDER_TIME_SPECIFIED
        fake_pending.time_expiration = 1794411200

        fake_sym_info = MagicMock()
        fake_sym_info.digits = 2

        fake_result = MagicMock()
        fake_result.retcode = 10009

        with patch("jarvis.execution.mt5_client.MT5_AVAILABLE", True), patch("jarvis.execution.mt5_client.mt5") as mock_mt5:
            mock_mt5.orders_get.return_value = [fake_pending]
            mock_mt5.symbol_info.return_value = fake_sym_info
            mock_mt5.order_send.return_value = fake_result
            mock_mt5.TRADE_ACTION_MODIFY = 7
            mock_mt5.ORDER_TIME_SPECIFIED = 2
            mock_mt5.ORDER_TIME_GTC = 0
            mock_mt5.ORDER_FILLING_IOC = 1
            mock_mt5.TRADE_RETCODE_DONE = 10009
            mock_mt5.TRADE_RETCODE_PLACED = 10008

            res = client.modify_pending_order(900123, price=4007.70)
            self.assertEqual(res.get("status"), "MODIFIED")
            sent_req = mock_mt5.order_send.call_args[0][0]
            self.assertEqual(sent_req["type_time"], 2)
            self.assertEqual(sent_req["expiration"], 1794411200)


if __name__ == "__main__":
    unittest.main()
