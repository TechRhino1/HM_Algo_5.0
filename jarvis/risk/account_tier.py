"""
HM Algo 2.0 — Unified Account Tier & Micro Definition System.

Provides a single source of truth for account sizing tiers, micro-mode gating,
lot caps, and economic expected-value (EV) scaling across all HM Algo 2.0 engines.
"""
from enum import Enum

class AccountTier(str, Enum):
    ULTRA_SURVIVAL = "ULTRA_SURVIVAL"  # < $40
    MICRO_GROWTH   = "MICRO_GROWTH"    # $40 - $100
    SMALL_ACCOUNT  = "SMALL_ACCOUNT"   # $100 - $250
    STANDARD       = "STANDARD"        # $250 - $1,000
    INSTITUTIONAL  = "INSTITUTIONAL"   # >= $1,000

def get_account_tier(equity: float) -> AccountTier:
    """Returns canonical AccountTier based on current account equity."""
    eq = float(equity)
    if eq < 40.0:
        return AccountTier.ULTRA_SURVIVAL
    elif eq < 100.0:
        return AccountTier.MICRO_GROWTH
    elif eq < 250.0:
        return AccountTier.SMALL_ACCOUNT
    elif eq < 1000.0:
        return AccountTier.STANDARD
    else:
        return AccountTier.INSTITUTIONAL

def is_micro_account(equity: float) -> bool:
    """True if account is in micro or small capital mode (< $500 equity)."""
    return float(equity) < 500.0

def is_nano_account(equity: float) -> bool:
    """True if account is in nano tier (< $250 equity)."""
    return float(equity) < 250.0

def get_nano_risk_cap(equity: float, symbol: str = "") -> tuple[float, float]:
    """
    Returns (max_risk_usd, max_risk_pct) allowed for the account tier and symbol.
    Unified source of truth for execution engine, sizing, and dashboard validation.
    """
    eq = float(equity)
    sym = (symbol or "").upper()
    is_gold = "XAU" in sym or "GOLD" in sym

    if eq < 250.0:
        return (6.00, 6.0) if is_gold else (5.00, 5.0)
    elif eq < 2500.0:
        return (28.00, 5.50) if is_gold else (12.00, 2.50)
    else:
        return (float("inf"), 1.0) if is_gold else (float("inf"), 0.50)

def get_milestone_risk_budget(equity: float, symbol: str = "") -> tuple[float, float]:
    """
    Returns (max_risk_usd, max_risk_pct) calibrated to the account growth milestone tier.
    Enforces invariant dollar risk for small/micro accounts ($100 - $1,000+).
    """
    eq = float(equity)
    sym = (symbol or "").upper()
    is_gold = "XAU" in sym or "GOLD" in sym

    if eq < 150.0:
        # Tier 1 ($100 - $150 Seed Sniper): strict $2.00-$2.50 risk budget
        return (3.00, 3.0) if is_gold else (2.50, 2.5)
    elif eq < 250.0:
        # Tier 2 ($150 - $250 Micro Compounder): $3.00-$4.50 risk budget
        return (5.00, 3.0) if is_gold else (4.00, 2.5)
    elif eq < 500.0:
        # Tier 3 ($250 - $500 Small Accelerator): $5.00-$8.00 risk budget
        return (8.00, 2.5) if is_gold else (7.00, 2.0)
    elif eq < 1000.0:
        # Tier 4 ($500 - $1,000 Standard Transition): $10.00-$15.00 risk budget
        return (15.00, 2.0) if is_gold else (12.00, 1.8)
    else:
        # Tier 5 (>= $1,000 Institutional): Percentage based
        return (float("inf"), 1.0) if is_gold else (float("inf"), 0.50)

def get_max_lot_cap(equity: float, symbol: str = "") -> float:
    """
    Returns hard lot cap for risk protection on smaller accounts.
    Allows 0.02 lots on tight Forex stops on $100 accounts while keeping
    Gold strictly capped at 0.01 lots until $150+ to protect against liquidation spikes.
    """
    eq = float(equity)
    sym = (symbol or "").upper()
    is_gold = "XAU" in sym or "GOLD" in sym

    if eq < 150.0:
        # Gold restricted to 0.01 lot floor on sub-$150 accounts; Forex allowed 0.02 lots on tight stops
        return 0.01 if is_gold else 0.02
    elif eq < 250.0:
        # Gold unlocked for 0.02 lot multi-target split; Forex allowed up to 0.03 lots
        return 0.02 if is_gold else 0.03
    elif eq < 500.0:
        return 0.05  # Enforce max 0.05 lots on $250-$500 micro accounts
    elif eq < 1000.0:
        return 0.10
    elif eq < 2500.0:
        return 0.15
    else:
        return 2.0  # Safe upper ceiling for standard retail accounts

def get_effective_min_ev(equity: float, planned_risk_dollars: float) -> float:
    """
    Returns mathematically sound expected-value hurdle (in dollars)
    scaled smoothly to account equity and 0.01 lot-floor constraints (§3).
    """
    eq = float(equity)
    prd = float(planned_risk_dollars)
    
    if eq < 100.0:
        return max(0.01, prd * 0.05)
    elif eq < 500.0:
        return max(0.05, prd * 0.10)
    elif eq < 1000.0:
        return max(0.10, prd * 0.20)
    else:
        return max(0.25, prd * 0.25)
