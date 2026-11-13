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
