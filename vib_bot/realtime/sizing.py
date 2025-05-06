#!/usr/bin/env python3
"""
realtime/sizing.py

Position sizing: risk %, volatility‑based sizing
"""
import logging

from vib_bot.config import RISK_PER_TRADE, STOP_LOSS_PCT, TAKE_PROFIT_PCT

logger = logging.getLogger("sizing")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


def compute_position_size(equity: float, entry_price: float) -> float:
    """
    Calculate how many units to buy/sell given:
      - equity: current account equity in USDT
      - entry_price: price per coin in USDT

    Uses RISK_PER_TRADE (e.g. 0.005 for 0.5% of equity)
    and STOP_LOSS_PCT (e.g. 0.012 for 1.2% stop loss).
    """
    risk_amount = equity * RISK_PER_TRADE
    per_unit_risk = entry_price * STOP_LOSS_PCT
    if per_unit_risk <= 0:
        logger.error("Invalid STOP_LOSS_PCT or entry_price → zero division")
        return 0.0

    qty = risk_amount / per_unit_risk
    logger.info(
        f"Sizing → equity={equity:.2f}, price={entry_price:.4f}, "
        f"risk_amt={risk_amount:.2f}, qty={qty:.6f}"
    )
    return qty


def compute_exit_levels(entry_price: float) -> tuple[float, float]:
    """
    Given an entry price, returns (stop_loss_price, take_profit_price)
      - stop_loss = entry_price * (1 - STOP_LOSS_PCT)
      - take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
    """
    sl = entry_price * (1.0 - STOP_LOSS_PCT)
    tp = entry_price * (1.0 + TAKE_PROFIT_PCT)
    logger.info(
        f"Exit levels → entry={entry_price:.4f}, SL={sl:.4f}, TP={tp:.4f}"
    )
    return sl, tp