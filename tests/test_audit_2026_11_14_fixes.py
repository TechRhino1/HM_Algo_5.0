"""Regressions for defects confirmed in the 2026-11-14 audit."""
from types import SimpleNamespace
from unittest.mock import patch

from jarvis.api.remote_auth import RemoteAuthEngine
from jarvis.risk.position_sizing import PositionSizer
from tests.test_live_readiness_fixes import _demo_client, _order_mt5


def test_filling_fallback_fill_does_not_send_a_duplicate_order():
    fake = _order_mt5(10009, 0.10)
    ok = SimpleNamespace(retcode=10009, deal=1, order=555, price=1.0852, volume=0.10, comment="ok")
    rejected = SimpleNamespace(retcode=10030, deal=0, order=0, price=0.0, volume=0.0, comment="filling")
    fake.order_send.side_effect = [rejected, ok, ok, ok, ok]  # IOC rejected, FOK fills
    with patch("jarvis.execution.mt5_client.mt5", fake), \
         patch("jarvis.execution.mt5_client.MT5_AVAILABLE", True):
        _demo_client().send_market_order("EURUSD", "BUY", 0.10, sl_price=1.0800, tp_price=1.0900)
    assert fake.order_send.call_count == 2


def test_volume_is_rounded_down_not_up():
    fake = _order_mt5(10009, 0.1)
    fake.symbol_info.return_value.volume_step = 0.1
    fake.symbol_info.return_value.volume_min = 0.1
    with patch("jarvis.execution.mt5_client.mt5", fake), \
         patch("jarvis.execution.mt5_client.MT5_AVAILABLE", True):
        _demo_client().send_market_order("EURUSD", "BUY", 0.19, sl_price=1.0800, tp_price=1.0900)
    assert fake.order_send.call_args[0][0]["volume"] == 0.1


def test_a_zero_halt_multiplier_sizes_to_zero_even_on_micro_accounts():
    for equity in (200.0, 5000.0):
        lots = PositionSizer.calculate_lot_size(
            account_balance=equity, entry_price=1.1000, sl_price=1.0950, risk_pct=0.5,
            symbol_info={"name": "EURUSD", "trade_contract_size": 100000,
                         "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01},
            model_confidence=0.9, portfolio_heat_multiplier=0.0,
        )
        assert lots == 0.0


def test_a_locked_out_user_cannot_log_in_even_with_the_right_password():
    pw = "correct horse battery staple"
    saved_users = dict(RemoteAuthEngine._users)
    saved_fail = {k: list(v) for k, v in RemoteAuthEngine._failed_attempts.items()}
    RemoteAuthEngine._users["locktest"] = {
        "username": "locktest", "role": "TRADER", "full_name": "x",
        "password_hash": RemoteAuthEngine._hash_password(pw), "created_at": 0.0,
    }
    try:
        for _ in range(RemoteAuthEngine._max_failed_attempts):
            RemoteAuthEngine.verify_credentials("locktest", "wrong", client_ip="203.0.113.9")
        user, err = RemoteAuthEngine.verify_credentials("locktest", pw, client_ip="203.0.113.9")
        assert user is None and "locked" in err.lower()
    finally:
        RemoteAuthEngine._users = saved_users
        RemoteAuthEngine._failed_attempts = saved_fail


def test_backtest_bar_clock_rolls_the_daily_loss_day_over():
    from datetime import datetime, timezone
    from jarvis.risk.risk_engine import RiskEngine

    engine = RiskEngine(is_backtest=True)
    day1 = datetime(2026, 7, 8, 15, tzinfo=timezone.utc).timestamp()
    day2 = datetime(2026, 7, 9, 15, tzinfo=timezone.utc).timestamp()
    engine.set_clock(lambda: day1)
    assert engine.drawdown_guard._today() == "2026-07-08"
    engine.set_clock(lambda: day2)
    assert engine.drawdown_guard._today() == "2026-07-09"


def test_gold_exit_runs_to_a_fixed_target_without_profit_locks():
    from jarvis.execution.exit_policy import ExitPolicy, evaluate_exit

    gold = ExitPolicy.for_symbol("XAUUSD", None, trade_style="SWING")
    assert gold.profit_locks is False and gold.milestones == []
    # +1.6R of a 2.5R target used to trip breakeven, partials and progress locks.
    dec = evaluate_exit(
        side="BUY", entry=4000.0, initial_sl=3982.0, current_sl=3982.0, tp=4045.0,
        price=4028.8, favorable_dist=28.8, atr=20.0, policy=gold,
    )
    assert dec.new_sl in (0.0, 3982.0) and dec.partial_close_pct == 0.0 and not dec.be_locked
    # Other symbols keep the managed exit.
    assert ExitPolicy.for_symbol("EURUSD", None, trade_style="SWING").profit_locks is True


def test_gold_order_take_profit_is_two_and_a_half_times_the_stop():
    from datetime import datetime, timezone
    from unittest.mock import MagicMock
    from jarvis.execution.execution_engine import ExecutionEngine
    from jarvis.data.schemas import DecisionObject, TradeQualityGateResult, RegimeOutput, MarketRegime

    mt5 = MagicMock()
    mt5.is_connected = True
    mt5.get_pending_orders.return_value = []
    mt5.place_pending_order.return_value = {"status": "PLACED", "ticket": 1, "price": 3984.94}
    state = MagicMock(is_safe_mode=False, is_circuit_broken=False, is_drawdown_limit_reached=False)
    state.account.equity = 10000.0
    state.get_market_context.return_value = None
    regime = RegimeOutput(primary_regime=MarketRegime.TREND_BEAR, probabilities={"TREND_BEAR": 0.9}, confidence=0.9)
    decision = DecisionObject(
        symbol="XAUUSD", timestamp=datetime.now(timezone.utc), regime=regime, bias="SELL",
        probabilities={"SELL": 0.8}, strategy="TREND_PULLBACK", entry_price=3984.94,
        stop_loss=4002.94, take_profit=3913.14, risk_reward_ratio=3.99, calculated_risk_percent=0.5,
        expected_value=1.0, model_confidence=0.6, adversarial_penalty=0.0, invalidation_levels=[],
        bull_case=[], bear_case=[], risk_factors=[], quality_gate=TradeQualityGateResult(passed=True, checks={}),
        decision="EXECUTE", execution_authorized=True, order_type="LIMIT", ai_score=90.0,
    )
    with patch("jarvis.data.database.TRADE_DB.log_trade"):
        ExecutionEngine(mt5_client=mt5, state_manager=state).execute_decision(decision, lots=0.01)
    reward, risk = decision.entry_price - decision.take_profit, decision.stop_loss - decision.entry_price
    assert abs(reward / risk - 2.5) < 0.01  # tick rounding only
    assert decision.risk_reward_ratio == 2.5
