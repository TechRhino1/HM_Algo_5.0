"""Regression tests for post-update live trade issues and pipeline improvements."""
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from jarvis.data.schemas import (
    AccountSnapshot,
    DecisionObject,
    LiquidityContext,
    MarketContext,
    MarketRegime,
    MomentumContext,
    PositionSnapshot,
    RegimeOutput,
    SessionContext,
    StructureContext,
    TradeQualityGateResult,
    VolatilityContext,
)
from jarvis.execution.exit_policy import ExitPolicy, evaluate_exit
from jarvis.execution.position_monitor import PositionMonitorEngine
from jarvis.intelligence.decision_engine import DecisionEngine
from jarvis.risk.circuit_breaker import CircuitBreaker
from jarvis.risk.loss_cooldown import LossCooldownManager
from jarvis.risk.risk_engine import RiskEngine


def _make_context(
    symbol: str = "XAUUSD",
    price: float = 4000.0,
    bid: float = 3999.9,
    ask: float = 4000.1,
    atr: float = 5.0,
    spread_pips: float = 2.0,
    trend_score: float = -30.0,
    choch: bool = False,
    choch_type: str = "NONE",
    struct_bias: str = "BEARISH",
    zone: str = "PREMIUM",
    is_prime: bool = True,
) -> MarketContext:
    return MarketContext(
        symbol=symbol,
        timestamp=datetime(2026, 11, 12, 1, 0, tzinfo=timezone.utc),
        current_price=price,
        bid=bid,
        ask=ask,
        structure=StructureContext(
            bias=struct_bias,
            discount_premium_zone=zone,
            choch=choch,
            choch_type=choch_type,
        ),
        liquidity=LiquidityContext(),
        volatility=VolatilityContext(atr=atr, current_spread_pips=spread_pips),
        momentum=MomentumContext(trend_score=trend_score, rsi=50.0, adx=25.0),
        session=SessionContext(current_session="NY" if is_prime else "SYDNEY_PACIFIC", is_prime_session=is_prime),
    )


class TestTimeStopAndExitPolicyFixes(unittest.TestCase):
    def test_range_regime_does_not_shrink_time_stop(self):
        trend_pol = ExitPolicy.for_symbol("GBPUSD", trade_style="DAY_TRADING", regime="TREND_BEAR")
        range_pol = ExitPolicy.for_symbol("GBPUSD", trade_style="DAY_TRADING", regime="RANGE")
        self.assertAlmostEqual(trend_pol.time_stop_minutes, 300.0, places=4)
        self.assertAlmostEqual(range_pol.time_stop_minutes, 300.0, places=4)

    def test_peak_mfe_immune_to_premature_time_stop(self):
        """Regression for #2372079602 GOLD.i#: +0.18R peak, +0.09R current must not TIME_STOP."""
        policy = ExitPolicy.for_symbol("XAUUSD", trade_style="SCALP")
        dec = evaluate_exit(
            side="SELL",
            entry=3989.9,
            initial_sl=4007.84,  # 1R = 17.94
            current_sl=4007.84,
            tp=3949.9,
            price=3988.3,  # +1.60 -> +0.089R
            favorable_dist=1.60,
            peak_favorable_dist=3.26,  # +0.182R peak (> 0.15R)
            atr=4.0,
            policy=policy,
            hold_minutes=95.0,
            partial_executable=False,
        )
        self.assertFalse(dec.should_close)


class TestCircuitBreakerScratchProtection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = f"{self.tmp.name}/cb.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_scratch_losses_do_not_trip_circuit_breaker_or_cooldown(self):
        cb = CircuitBreaker(consecutive_loss_limit=5, db_path=self.db_path)
        lcd = LossCooldownManager()
        # Simulate 5 tiny -0.02R .. -0.06R TIME_STOP scratches
        for _ in range(5):
            cb.record_trade_result(False, symbol="BTCUSD", regime="TREND_BULL", r_multiple=-0.04)
            lcd.record_trade_result(-0.30, False, symbol="BTCUSD", r_multiple=-0.04)
        self.assertFalse(cb.is_tripped)
        self.assertEqual(cb.consecutive_losses, 0)
        self.assertEqual(lcd.consecutive_losses, 0)

        # Real stop-outs (< -0.15R) still trip velocity limit
        for _ in range(3):
            cb.record_trade_result(False, symbol="ETHUSD", regime="TREND_BEAR", r_multiple=-0.65)
        self.assertTrue(cb.is_tripped)


class TestPositionMonitorFixes(unittest.TestCase):
    def test_fresh_limit_order_not_cancelled_within_90s(self):
        """Regression for ETHUSD #2372124509/#2372132569 1s self-cancellation."""
        mock_mt5 = MagicMock()
        mock_mt5.get_broker_time.return_value = 1000001.0
        mock_mt5.get_pending_orders.return_value = [{
            "ticket": 2372124509,
            "symbol": "ETHUSD#",
            "type": "SELL_LIMIT",
            "price_open": 2082.68,
            "sl": 2112.29,
            "tp": 2000.19,
            "comment": "HMA2_DAY_TREN_LIVE",
            "time_setup": 1000000.0,  # age = 1s
        }]
        state = MagicMock(positions=[])
        mon = PositionMonitorEngine(mt5_client=mock_mt5, data_feed=MagicMock(), context_engine=MagicMock(), state_manager=state)
        ctx = _make_context(
            symbol="ETHUSD", price=2077.0, bid=2076.5, ask=2077.5, atr=10.0,
            spread_pips=100.0, trend_score=55.0, choch=True, choch_type="BULLISH", struct_bias="BULLISH",
        )
        mon._get_context = MagicMock(return_value=ctx)
        mon._manage_pending_orders_tick()
        mock_mt5.cancel_pending_order.assert_not_called()

    def test_underwater_regime_inversion_does_not_halve_sl(self):
        """Regression for ETHUSD/SOLUSD REGIME_DRAWDOWN_TIGHTEN SL halving & 55-line spam."""
        mon = PositionMonitorEngine(mt5_client=MagicMock(), data_feed=MagicMock(), context_engine=MagicMock(), state_manager=MagicMock())
        pos = PositionSnapshot(
            ticket=2372096296, symbol="ETHUSD#", type="BUY", volume=0.1,
            open_price=2091.32, current_price=2087.0, sl=2069.50, tp=2145.87,
            profit=-0.43, swap=0.0, commission=0.0, open_time=datetime.now(timezone.utc).isoformat(), magic=1,
        )
        mon._initial_risk_dist[pos.ticket] = 21.82
        ctx = _make_context(symbol="ETHUSD", price=2087.0, bid=2086.5, ask=2087.5, atr=10.0)
        regime = RegimeOutput(primary_regime=MarketRegime.TREND_BEAR, probabilities={"TREND_BEAR": 0.81}, confidence=0.81)
        new_sl, act = mon._check_regime_invalidation(pos, ctx, regime, 2087.0, 10.0, pos.sl, digits=2, pip_size=0.1)
        self.assertEqual(new_sl, pos.sl)
        self.assertIsNone(act)

    def test_buy_limit_nudge_anchors_to_ask_not_bid(self):
        """Regression for USDJPY #2372127157 0.1p miss when BUY_LIMIT clamped below Bid."""
        mock_mt5 = MagicMock()
        mock_mt5.get_broker_time.return_value = 1000100.0
        mock_mt5.modify_pending_order.return_value = {"status": "MODIFIED"}
        mock_mt5.get_pending_orders.return_value = [{
            "ticket": 2372127157,
            "symbol": "USDJPY#",
            "type": "BUY_LIMIT",
            "price_open": 159.864,
            "sl": 159.790,
            "tp": 160.049,
            "comment": "HMA2_SCLP_RANG_LIVE",
            "time_setup": 1000000.0,
        }]
        mon = PositionMonitorEngine(mt5_client=mock_mt5, data_feed=MagicMock(), context_engine=MagicMock(), state_manager=MagicMock(positions=[]))
        # Tick 1: Ask touches 159.872 (gap 0.008 <= tol 0.040), Bid is 159.862
        ctx1 = _make_context(symbol="USDJPY", price=159.867, bid=159.862, ask=159.872, atr=0.30, spread_pips=1.0, struct_bias="BULLISH")
        # Tick 2: Price ticks up slightly -> bounce retrace triggers BUY_LIMIT nudge closer to Ask
        ctx2 = _make_context(symbol="USDJPY", price=159.879, bid=159.874, ask=159.884, atr=0.30, spread_pips=1.0, struct_bias="BULLISH")
        mon._get_context = MagicMock(side_effect=[ctx1, ctx2])
        mon._manage_pending_orders_tick()
        mock_mt5.modify_pending_order.assert_not_called()
        mon._manage_pending_orders_tick()
        mock_mt5.modify_pending_order.assert_called_once()
        nudged = mock_mt5.modify_pending_order.call_args.kwargs["price"]
        # Old Bid-clamp (bid_val - retrace = 159.869) capped below Ask fill zone;
        # new Ask-clamp nudges cleanly up to 159.875 (< Ask 159.884).
        self.assertGreater(nudged, 159.870)
        self.assertLess(nudged, 159.884)


class TestSARAndOffHoursGuards(unittest.TestCase):
    def test_sar_rejects_low_confidence_and_limit_flip_even_with_override(self):
        from jarvis.application.orchestrator import JarvisOrchestrator
        with patch("jarvis.application.orchestrator.MT5Client"), \
             patch("jarvis.application.orchestrator.ExecutionEngine"):
            orch = JarvisOrchestrator(symbols=["USDJPY"], mode="paper", trade_style="SCALP")
            orch.mt5_client.close_position = MagicMock()
            pos = PositionSnapshot(
                ticket=2372134152, symbol="USDJPY#", type="BUY", volume=0.08,
                open_price=159.872, current_price=159.830, sl=159.789, tp=160.080,
                profit=-2.10, swap=0.0, commission=0.0, open_time=datetime.now(timezone.utc).isoformat(), magic=1,
            )
            dec = MagicMock(
                bias="SELL", decision="EXECUTE", trade_style="SCALP", order_type="MARKET",
                model_confidence=0.30, expected_value=0.47, risk_reward_ratio=2.37, adversarial_penalty=36.0,
            )
            ctx = _make_context(symbol="USDJPY", price=159.830, is_prime=False)
            _, _, flipped = orch._apply_stop_and_reverse_if_eligible(
                dec, ctx, [pos], [pos], "USDJPY", entry_override=True
            )
            self.assertFalse(flipped)
            orch.mt5_client.close_position.assert_not_called()

    def test_risk_engine_blocks_off_hours_low_conf_or_high_adv(self):
        re = RiskEngine(max_risk_per_trade_pct=1.0)
        acc = AccountSnapshot(login=169172945, server="XMGlobal-MT5 2", leverage=1000, balance=840.0, equity=840.0, margin=0.0, free_margin=840.0, margin_level=9999.0, trade_allowed=True)
        dec = DecisionObject(
            symbol="USDJPY",
            timestamp=datetime.now(timezone.utc),
            regime=RegimeOutput(primary_regime=MarketRegime.REVERSAL, probabilities={}, confidence=0.40),
            bias="SELL",
            probabilities={"BUY": 0.30, "SELL": 0.50, "HOLD": 0.20},
            strategy="CHOCH_STRUCTURAL_REVERSAL",
            trade_style="SCALP",
            order_type="MARKET",
            decision="EXECUTE",
            entry_price=159.83,
            stop_loss=159.91,
            take_profit=159.63,
            risk_reward_ratio=2.5,
            calculated_risk_percent=1.0,
            expected_value=0.47,
            model_confidence=0.30,
            adversarial_penalty=36.0,
            invalidation_levels=[],
            bull_case=[],
            bear_case=[],
            risk_factors=[],
            quality_gate=TradeQualityGateResult(passed=True, checks={}, failing_reasons=[]),
        )
        ctx = _make_context(symbol="USDJPY", price=159.83, bid=159.825, ask=159.835, atr=0.10, spread_pips=1.0, is_prime=False)
        res = re.authorize_execution(
            dec, acc, [],
            {"name": "USDJPY#", "trade_contract_size": 100_000.0, "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01},
            current_spread_pips=1.0, max_allowed_spread_pips=4.0, context=ctx, entry_authorized_override=True,
        )
        self.assertFalse(res["authorized"])
        self.assertTrue(any("OFF_HOURS_QUALITY_GATE" in r for r in res["reasons"]))

    def test_sar_aborts_if_pre_authorize_rejects(self):
        from jarvis.application.orchestrator import JarvisOrchestrator
        with patch("jarvis.application.orchestrator.MT5Client"), \
             patch("jarvis.application.orchestrator.ExecutionEngine"):
            orch = JarvisOrchestrator(symbols=["XAUUSD"], mode="paper", trade_style="DAY_TRADING")
            orch.mt5_client.close_position = MagicMock()
            pos = PositionSnapshot(
                ticket=999111, symbol="XAUUSD", type="BUY", volume=0.01,
                open_price=4000.0, current_price=4010.0, sl=3980.0, tp=4050.0,
                profit=10.0, swap=0.0, commission=0.0, open_time=datetime.now(timezone.utc).isoformat(), magic=1,
            )
            dec = MagicMock(
                bias="SELL", decision="EXECUTE", trade_style="DAY_TRADING", order_type="MARKET",
                model_confidence=0.72, expected_value=1.5, risk_reward_ratio=2.4, adversarial_penalty=10.0,
            )
            ctx = _make_context(symbol="XAUUSD", price=4010.0, is_prime=True)
            _, _, flipped = orch._apply_stop_and_reverse_if_eligible(
                dec, ctx, [pos], [pos], "XAUUSD", entry_override=False, pre_authorize_fn=lambda _sim: False
            )
            self.assertFalse(flipped)
            orch.mt5_client.close_position.assert_not_called()

    def test_off_hours_fx_limit_clamps_tighter_than_prime(self):
        eng = DecisionEngine()
        ctx_off = _make_context(
            symbol="EURUSD", price=1.10910, bid=1.10905, ask=1.10915,
            atr=0.00120, spread_pips=1.0, trend_score=-60.0, zone="DISCOUNT", is_prime=False,
        )
        e_off, *_ = eng._apply_limit_entry_geometry(
            context=ctx_off, bias="SELL", style_str="DAY_TRADING", cur_mkt_price=1.10910,
            entry_price=1.10910, sl_price=1.11500, tp_price=1.09500,
            risk_dist=0.00590, tp_dist=0.01410, atr_val=0.00120, spread_dist=0.00010,
            pip_size=0.0001, digits=5, tp1_price=None, tp2_price=None, tp3_price=None,
            first_target_price=None, use_structure_target=True,
        )
        # With no FVG/OB, offset = 0.30*ATR = 3.6p, capped by off-hours 0.40*ATR = 4.8p
        # Also if OB sits far above (say +9p), capped at <= +4.8p instead of +9.6p!
        ctx_ob = _make_context(
            symbol="EURUSD", price=1.10910, bid=1.10905, ask=1.10915,
            atr=0.00120, spread_pips=1.0, trend_score=-60.0, zone="DISCOUNT", is_prime=False,
        )
        ctx_ob.structure.order_blocks = [{"type": "BEARISH_OB", "low": 1.11000, "high": 1.11030}]
        e_capped, *_ = eng._apply_limit_entry_geometry(
            context=ctx_ob, bias="SELL", style_str="DAY_TRADING", cur_mkt_price=1.10910,
            entry_price=1.10910, sl_price=1.11500, tp_price=1.09500,
            risk_dist=0.00590, tp_dist=0.01410, atr_val=0.00120, spread_dist=0.00010,
            pip_size=0.0001, digits=5, tp1_price=None, tp2_price=None, tp3_price=None,
            first_target_price=None, use_structure_target=True,
        )
        self.assertLessEqual(e_capped - 1.10910, 0.00050)
        self.assertGreater(e_off, 1.10910)

    def test_recover_pending_features_preserves_trade_style(self):
        from jarvis.application.orchestrator import JarvisOrchestrator
        with patch("jarvis.application.orchestrator.MT5Client"), \
             patch("jarvis.application.orchestrator.ExecutionEngine"):
            orch = JarvisOrchestrator(symbols=["USDCAD"], mode="paper", trade_style="DAY_TRADING")
            orch.trade_memory = MagicMock()
            orch.trade_memory.fetch_trade.return_value = {
                "ticket": 2371998797, "symbol": "USDCAD", "trade_type": "BUY",
                "entry_price": 1.42451, "sl": 1.42004, "strategy": "TREND_PULLBACK",
                "regime": "TREND_BULL", "reasoning": '{"trade_style": "DAY_TRADING"}',
                "ml_features": "[0.5, 0.2]",
            }
            rec = orch._recover_pending_features(2371998797)
            self.assertEqual(rec["trade_style"], "DAY_TRADING")


if __name__ == "__main__":
    unittest.main()
