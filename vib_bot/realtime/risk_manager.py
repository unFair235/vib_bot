#!/usr/bin/env python3
"""
realtime/risk_manager.py

Stop‑loss / take‑profit computation and portfolio drawdown breaker.
"""
import logging
from typing import Tuple

from vib_bot.config import STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_DAILY_DRAWDOWN
from vib_bot.realtime.trade_tracker import get_equity_curve
from vib_bot.utils.metrics import compute_drawdowns

logger = logging.getLogger("risk_manager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


def assess_risk(
    action: str,
    entry_price: float,
    features: dict
) -> Tuple[bool, float, float]:
    """
    Given an action ("LONG" or "SHORT") and entry price, compute stop-loss and take-profit levels,
    then enforce a maximum portfolio drawdown rule.
    Returns a tuple (ok: bool, stop_loss: float, take_profit: float).
    """
    action = action.upper()
    # 1) Compute stop-loss and take-profit
    if action == "LONG":
        stop_loss   = entry_price * (1.0 - STOP_LOSS_PCT)
        take_profit = entry_price * (1.0 + TAKE_PROFIT_PCT)
    elif action == "SHORT":
        stop_loss   = entry_price * (1.0 + STOP_LOSS_PCT)
        take_profit = entry_price * (1.0 - TAKE_PROFIT_PCT)
    else:
        logger.error(f"Unknown action '{action}' in risk check")
        return False, 0.0, 0.0

    # 2) Portfolio drawdown breaker
    try:
        equity_curve = get_equity_curve()  # array-like of historical equity
        dd_metrics   = compute_drawdowns(equity_curve)
        max_dd       = dd_metrics["max_drawdown"]
        if max_dd < -MAX_DAILY_DRAWDOWN:
            logger.warning(
                f"Portfolio drawdown {max_dd:.2%} exceeds limit "
                f"{MAX_DAILY_DRAWDOWN:.2%}; aborting trade."
            )
            return False, stop_loss, take_profit
    except Exception as e:
        logger.error(f"Error computing drawdown: {e}; allowing trade by default")

    # 3) Passed all checks
    return True, stop_loss, take_profit