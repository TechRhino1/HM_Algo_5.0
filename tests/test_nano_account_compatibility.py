"""
Unit test suite verifying 100-dollar Nano Account Compatibility across all risk,
sizing, execution, arbiter, and exposure engines.
"""
import pytest
from jarvis.risk.account_tier import is_nano_account, get_nano_risk_cap, get_max_lot_cap
from jarvis.risk.position_sizing import PositionSizer
from jarvis.risk.drawdown import DrawdownGuard
from jarvis.risk.exposure import ExposureManager
from jarvis.data.schemas import AccountSnapshot, PositionSnapshot, DecisionObject, MarketContext
from jarvis.intelligence.opportunity_arbiter import UniversalOpportunityArbiter
from jarvis.api.server import JarvisRequestHandler

def test_account_tier_nano_detection():
    assert is_nano_account(50.0) is True
    assert is_nano_account(100.0) is True
    assert is_nano_account(249.99) is True
    assert is_nano_account(250.0) is False
    assert is_nano_account(850.0) is False

def test_nano_risk_caps():
    # Nano tier (< $250)
    usd, pct = get_nano_risk_cap(100.0, "EURUSD")
    assert usd == 5.00
    assert pct == 5.0

    usd, pct = get_nano_risk_cap(100.0, "XAUUSD")
    assert usd == 6.00
    assert pct == 6.0

    # Micro tier ($250 - $2500)
    usd, pct = get_nano_risk_cap(850.0, "EURUSD")
    assert usd == 12.00
    assert pct == 2.50

    usd, pct = get_nano_risk_cap(850.0, "XAUUSD")
    assert usd == 28.00
    assert pct == 5.50

    # Max lot cap
    assert get_max_lot_cap(50.0) == 0.01
    assert get_max_lot_cap(100.0) == 0.01
    assert get_max_lot_cap(150.0) == 0.01
    assert get_max_lot_cap(200.0) == 0.02
    assert get_max_lot_cap(850.0) == 0.10

def test_position_sizing_on_100_dollar_account():
    eurusd_info = {
        "name": "EURUSD",
        "canonical": "EURUSD",
        "point": 0.00001,
        "contract_size": 100000.0,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01
    }
    
    # 1. Standard EURUSD 30-pip stop: entry=1.08500, sl=1.08200 (distance=0.00300)
    # Risk for 0.01 lot = 0.01 * 100,000 * 0.00300 = $3.00 (3.0% of $100)
    # Allowed under nano tier (risk <= $5.00, <= 5.0% and <= $3.50 floor)
    lots = PositionSizer.calculate_lot_size(
        account_balance=100.0,
        entry_price=1.08500,
        sl_price=1.08200,
        risk_pct=0.50,
        symbol_info=eurusd_info
    )
    assert lots == 0.01

    # 2. Wide EURUSD 70-pip stop: entry=1.08500, sl=1.07800 (distance=0.00700)
    # Risk for 0.01 lot = 0.01 * 100,000 * 0.00700 = $7.00 (7.0% of $100)
    # Rejected because $7.00 > $5.00 ceiling
    lots_wide = PositionSizer.calculate_lot_size(
        account_balance=100.0,
        entry_price=1.08500,
        sl_price=1.07800,
        risk_pct=0.50,
        symbol_info=eurusd_info
    )
    assert lots_wide == 0.0

def test_gold_sizing_on_100_dollar_account():
    xau_info = {
        "name": "XAUUSD",
        "canonical": "XAUUSD",
        "point": 0.01,
        "contract_size": 100.0,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01
    }
    # Tight gold scalp: $5.00 stop distance (e.g. 2700.00 to 2695.00)
    # Risk for 0.01 lot = 0.01 * 100 * 5.00 = $5.00
    # Allowed on nano account (<= $6.00 gold ceiling)
    lots_gold_tight = PositionSizer.calculate_lot_size(
        account_balance=100.0,
        entry_price=2700.00,
        sl_price=2695.00,
        risk_pct=0.50,
        symbol_info=xau_info
    )
    assert lots_gold_tight == 0.01

    # Wide gold stop: $12.00 stop distance (2700.00 to 2688.00)
    # Risk for 0.01 lot = 0.01 * 100 * 12.00 = $12.00
    # Rejected because $12.00 > $6.00 gold ceiling
    lots_gold_wide = PositionSizer.calculate_lot_size(
        account_balance=100.0,
        entry_price=2700.00,
        sl_price=2688.00,
        risk_pct=0.50,
        symbol_info=xau_info
    )
    assert lots_gold_wide == 0.0

def test_drawdown_guard_nano_account():
    guard = DrawdownGuard(db_path="", max_daily_loss_pct=4.0, max_total_drawdown_pct=10.0)
    guard.daily_start_equity = 100.0
    guard.peak_equity = 100.0

    # 1. Single loss of $3.50 (3.5% loss):
    # Standard 4.0% cap would be close to failing, but nano cap is 6.0%.
    res = guard.check_limits(current_equity=96.50, current_balance=96.50)
    assert res["passed"] is True
    assert res["daily_loss_pct"] == 3.5

    # 2. Cumulative loss of $7.00 (7.0% loss):
    # Breaches effective 6.0% daily cap
    res_breach = guard.check_limits(current_equity=93.00, current_balance=93.00)
    assert res_breach["passed"] is False
    assert any("Max Daily Loss breached" in b for b in res_breach["breaches"])

def test_exposure_manager_nano_account():
    exp = ExposureManager(max_open_positions=3)
    acct = AccountSnapshot(
        login=123456,
        server="XMGlobal-MT5",
        balance=100.0,
        equity=100.0,
        margin=0.0,
        free_margin=100.0,
        margin_level=0.0,
        leverage=500,
        currency="USD"
    )
    positions = [
        PositionSnapshot(
            ticket=1,
            symbol="EURUSD",
            type="BUY",
            volume=0.01,
            open_price=1.0850,
            current_price=1.0850,
            sl=1.0820,
            tp=1.0910,
            profit=0.0,
            swap=0.0,
            commission=0.0,
            open_time="2026-11-10 20:00:00",
            magic=100001
        )
    ]
    # Check 2nd position (total 2) -> Allowed
    res1 = exp.check_exposure(
        symbol="GBPUSD",
        positions=positions,
        account=acct,
        incoming_risk_usd=3.00
    )
    assert res1["passed"] is True

    # Check 3rd position (total 3) -> Blocked for nano account (< $250 equity has max 2 positions)
    positions.append(
        PositionSnapshot(
            ticket=2,
            symbol="GBPUSD",
            type="BUY",
            volume=0.01,
            open_price=1.2700,
            current_price=1.2700,
            sl=1.2670,
            tp=1.2760,
            profit=0.0,
            swap=0.0,
            commission=0.0,
            open_time="2026-11-10 20:00:00",
            magic=100001
        )
    )
    res2 = exp.check_exposure(
        symbol="USDJPY",
        positions=positions,
        account=acct,
        incoming_risk_usd=3.00
    )
    assert res2["passed"] is False
    assert any("Max Concurrent Positions reached" in b for b in res2["breaches"])

def test_dashboard_manual_risk_nano_validation():
    # On $100 equity:
    # 0.01 lots EURUSD with 30-pip stop ($3.00 risk) should pass
    class DummyHandler:
        pass
    dummy = DummyHandler()
    dummy.mt5_client = None
    dummy.orchestrator = None

    err_ok = JarvisRequestHandler._validate_trade_risk(
        dummy,
        symbol="EURUSD",
        price=1.0850,
        sl=1.0820,
        lots=0.01,
        equity=100.0
    )
    assert err_ok is None

    # 0.01 lots EURUSD with 70-pip stop ($7.00 risk) should be rejected
    err_wide = JarvisRequestHandler._validate_trade_risk(
        dummy,
        symbol="EURUSD",
        price=1.0850,
        sl=1.0780,
        lots=0.01,
        equity=100.0
    )
    assert err_wide is not None
    assert "Trade risk $7.00" in err_wide
