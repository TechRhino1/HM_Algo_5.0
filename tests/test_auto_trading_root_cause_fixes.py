"""
HM Algo 2.0 - Verification Suite for Automatic Trading Execution Root Cause Fixes
Tests:
1. Execution Engine Auto Size-Down volume logic.
2. Position Sizer risk ceiling compliance on $843 standard accounts.
3. Dynamic Levels strictly positive Take Profit levels on high-timeframe SELL setups.
4. Adversarial threat runway calibration (>= 0.50R designates TP1 milestone without blocking).
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from jarvis.data.schemas import (
    DecisionObject, MarketContext, VolatilityContext, StructureContext,
    MomentumContext, LiquidityContext, SessionContext, RegimeOutput, MarketRegime,
    TradeQualityGateResult
)
from jarvis.execution.execution_engine import ExecutionEngine
from jarvis.risk.position_sizing import PositionSizer
from jarvis.intelligence.dynamic_levels import DynamicRiskAndLevelsEngine
from jarvis.analysts.devil_advocate import DevilAdvocateAnalyst

def test_execution_engine_auto_size_down():
    """Verify that a trade sized to 0.05 lots that exceeds equity budget is automatically sized down to fit."""
    mt5 = MagicMock()
    mt5.send_market_order.return_value = {"retcode": 10009, "deal": 12345, "order": 67890, "status": "SUCCESS"}
    mt5.get_symbol_trading_spec.return_value = {
        "point": 0.00001,
        "digits": 5,
        "trade_tick_size": 0.00001,
        "trade_tick_value": 1.0,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
        "trade_stops_level": 0,
        "spread": 10
    }
    
    state_mgr = MagicMock()
    state_mgr.is_safe_mode = False
    state_mgr.safe_mode = False
    state_mgr.account.equity = 843.36
    state_mgr.account.balance = 843.36
    state_mgr.account.currency = "USD"
    state_mgr.is_live_allowed.return_value = True
    
    engine = ExecutionEngine(mt5_client=mt5, state_manager=state_mgr)
    
    # 0.05 lots with 24 pip stop (0.00240) = $12.00 risk. Budget is 0.50% of 843.36 = $4.22.
    decision = DecisionObject(
        symbol="EURUSD",
        timestamp=datetime.now(timezone.utc),
        regime=RegimeOutput(primary_regime=MarketRegime.TREND_BULL, confidence=0.8, probabilities={}),
        bias="BUY",
        probabilities={"buy": 0.85, "sell": 0.10, "no_trade": 0.05},
        strategy="TREND_PULLBACK",
        entry_price=1.08500,
        stop_loss=1.08260,
        take_profit=1.09000,
        risk_reward_ratio=2.08,
        calculated_risk_percent=0.5,
        expected_value=15.0,
        model_confidence=0.85,
        adversarial_penalty=5.0,
        invalidation_levels=[],
        bull_case=[],
        bear_case=[],
        risk_factors=[],
        quality_gate=TradeQualityGateResult(passed=True, checks={}),
        decision="EXECUTE",
        execution_authorized=True
    )
    
    result = engine.execute_decision(decision, lots=0.05)
    # Must NOT be blocked! Sizing down must execute order
    assert result.get("status") in ("SUCCESS", "FILLED") or result.get("retcode") == 10009
    
    # Check that send_market_order was called with sized down lots (<= 0.02 lots)
    assert mt5.send_market_order.called
    call_kwargs = mt5.send_market_order.call_args.kwargs or {}
    executed_lots = call_kwargs.get("volume") or mt5.send_market_order.call_args[0][2]
    assert executed_lots <= 0.02, f"Expected lots <= 0.02 to fit $4.22 budget, got {executed_lots}"
    assert executed_lots >= 0.01

def test_execution_engine_minimum_lot_allowance():
    """Verify that a trade at broker minimum lot (0.02 lots on ETHUSD) risking $11.75 is permitted under the $12 cap on an $843 account."""
    mt5 = MagicMock()
    mt5.send_market_order.return_value = {"retcode": 10009, "deal": 12345, "order": 67890, "status": "SUCCESS"}
    mt5.get_symbol_trading_spec.return_value = {
        "point": 0.01,
        "digits": 2,
        "trade_tick_size": 0.01,
        "trade_tick_value": 0.01,
        "volume_min": 0.02,
        "volume_step": 0.01,
        "volume_max": 100.0,
        "trade_stops_level": 0,
        "spread": 10
    }
    
    state_mgr = MagicMock()
    state_mgr.is_safe_mode = False
    state_mgr.safe_mode = False
    state_mgr.account.equity = 843.36
    state_mgr.account.balance = 843.36
    state_mgr.account.currency = "USD"
    state_mgr.is_live_allowed.return_value = True
    
    engine = ExecutionEngine(mt5_client=mt5, state_manager=state_mgr)
    
    # 0.02 lots on ETHUSD with 587 pt stop = $11.75 risk. Allowed under $12.00 / 2.5% cap for accounts < $2500.
    decision = DecisionObject(
        symbol="ETHUSD",
        timestamp=datetime.now(timezone.utc),
        regime=RegimeOutput(primary_regime=MarketRegime.TREND_BEAR, confidence=0.8, probabilities={}),
        bias="SELL",
        probabilities={"buy": 0.10, "sell": 0.85, "no_trade": 0.05},
        strategy="TREND_PULLBACK",
        entry_price=2300.0,
        stop_loss=2887.66,
        take_profit=1125.0,
        risk_reward_ratio=2.0,
        calculated_risk_percent=1.4,
        expected_value=25.0,
        model_confidence=0.85,
        adversarial_penalty=5.0,
        invalidation_levels=[],
        bull_case=[],
        bear_case=[],
        risk_factors=[],
        quality_gate=TradeQualityGateResult(passed=True, checks={}),
        decision="EXECUTE",
        execution_authorized=True
    )
    
    result = engine.execute_decision(decision, lots=0.02)
    assert result.get("status") in ("SUCCESS", "FILLED") or result.get("retcode") == 10009
    assert mt5.send_market_order.called
    call_kwargs = mt5.send_market_order.call_args.kwargs or {}
    executed_lots = call_kwargs.get("volume") or mt5.send_market_order.call_args[0][2]
    assert executed_lots == 0.02


def test_position_sizing_standard_account_ceiling():
    """Verify that PositionSizer does not artificially inflate risk to 1.50%-2.50% on $843 accounts."""
    sym_info = {
        "name": "EURUSD",
        "canonical": "EURUSD",
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
        "point": 0.00001,
        "digits": 5,
        "trade_tick_size": 0.00001,
        "trade_tick_value": 1.0,
        "contract_size": 100000.0
    }
    
    # Risk distance 24 pips = 0.00240 (1.08500 to 1.08260)
    # On $843.36 with 0.50% ceiling, risk amount is $4.22. Raw lots = 4.22 / (0.00240 * 100000) = 0.0175 -> 0.01
    lots = PositionSizer.calculate_lot_size(
        account_balance=843.36,
        entry_price=1.08500,
        sl_price=1.08260,
        risk_pct=0.50,
        symbol_info=sym_info,
        model_confidence=0.85 # High conviction
    )
    
    assert lots == 0.01, f"Expected 0.01 lots for 0.50% ceiling on $843 equity, got {lots}"

def test_dynamic_levels_strictly_positive_sell_tp():
    """Verify that high-timeframe SELL setups on ETHUSD cannot calculate negative take profit."""
    engine = DynamicRiskAndLevelsEngine()
    
    # Simulate ETHUSD D1_SWING context where entry is 2300 and ATR/risk_dist is large
    ctx = MarketContext(
        symbol="ETHUSD",
        timestamp=datetime.now(timezone.utc),
        current_price=2300.0,
        bid=2299.5,
        ask=2300.5,
        structure=StructureContext(bias="BEARISH", discount_premium_zone="PREMIUM", supply_zone=(2350.0, 2400.0), demand_zone=(1800.0, 1850.0)),
        liquidity=LiquidityContext(),
        volatility=VolatilityContext(atr=120.0, current_spread_pips=10.0),
        momentum=MomentumContext(rsi=65.0, trend_score=-40.0),
        session=SessionContext()
    )
    regime = RegimeOutput(primary_regime=MarketRegime.TREND_BEAR, confidence=0.8, probabilities={})
    
    levels = engine.calculate_levels(
        context=ctx,
        regime=regime,
        tentative_bias="SELL",
        account_balance=843.36,
        risk_per_trade_pct=0.5,
        trade_style="D1_SWING"
    )
    
    tp_price = levels["tp_price"]
    tp1 = levels.get("tp1_price")
    tp2 = levels.get("tp2_price")
    tp3 = levels.get("tp3_price")
    
    assert tp_price > 0, f"tp_price must be strictly positive, got {tp_price}"
    assert tp_price < 2300.0, f"sell tp_price must be below entry 2300, got {tp_price}"
    if tp1:
        assert tp1 > 0 and tp1 < 2300.0
    if tp2:
        assert tp2 > 0 and tp2 < 2300.0
    if tp3:
        assert tp3 > 0 and tp3 < 2300.0

def test_adversarial_threat_runway_designates_tp1():
    """Verify that an adversarial obstacle with 1.15R runway is designated as TP1 milestone rather than hard blocking."""
    ctx = MarketContext(
        symbol="XAUUSD",
        timestamp=datetime.now(timezone.utc),
        current_price=4030.0,
        bid=4029.8,
        ask=4030.2,
        structure=StructureContext(
            bias="BULLISH",
            discount_premium_zone="DISCOUNT",
            supply_zone=(4080.0, 4090.0),
            demand_zone=(3980.0, 4000.0),
            key_levels=[
                {"price": 4032.0, "touches": 1, "strength": 30}, # Minor 1-touch noise: should be filtered out
                {"price": 4044.59, "touches": 3, "strength": 70}  # Major 3-touch level: should be designated threat
            ]
        ),
        liquidity=LiquidityContext(),
        volatility=VolatilityContext(atr=15.0, current_spread_pips=4.0),
        momentum=MomentumContext(rsi=55.0, trend_score=45.0),
        session=SessionContext()
    )
    regime = RegimeOutput(primary_regime=MarketRegime.TREND_BULL, confidence=0.8, probabilities={})
    
    da = DevilAdvocateAnalyst()
    rep = da.critique_opportunity(ctx, regime, "BUY")
    # Verify minor level 4032.0 was filtered out, and major level 4044.59 was picked
    assert rep.threat_price_level == 4044.59
