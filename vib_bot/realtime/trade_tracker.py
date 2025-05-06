#!/usr/bin/env python3
"""
realtime/trade_tracker.py

Log every live trade + context for feedback,
and provide simple equity tracking for risk checks.
"""
import os
import json
import sqlite3
import logging
from typing import List, Optional

from vib_bot.config import MASTER_DB_FILE

# ─── logger setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger("trade_tracker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# ─── ensure table exists ──────────────────────────────────────────────────────
def _ensure_table():
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS executed_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            model_id TEXT,
            predicted_label INTEGER,
            action TEXT,
            price REAL,
            quantity REAL,
            stop_loss REAL,
            take_profit REAL,
            reason TEXT,
            features TEXT
        );
    """)
    conn.commit()
    conn.close()

_ensure_table()

# ─── log a trade ──────────────────────────────────────────────────────────────
def log_trade(
    timestamp: str,
    model_id: str,
    predicted_label: int,
    action: str,
    price: float,
    quantity: float,
    stop_loss: float,
    take_profit: float,
    reason: str,
    features: dict,
    symbol: Optional[str] = None
) -> None:
    """
    Insert an executed trade record into the executed_trades table.
    """
    try:
        conn = sqlite3.connect(MASTER_DB_FILE)
        conn.execute(
            """
            INSERT INTO executed_trades (
                timestamp,
                symbol,
                model_id,
                predicted_label,
                action,
                price,
                quantity,
                stop_loss,
                take_profit,
                reason,
                features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                symbol,
                model_id,
                predicted_label,
                action,
                price,
                quantity,
                stop_loss,
                take_profit,
                reason,
                json.dumps(features)
            )
        )
        conn.commit()
        logger.info(f"Logged trade {action} {quantity}@{price:.5f} for symbol {symbol}")
    except Exception as e:
        logger.error(f"Error logging trade: {e}")
    finally:
        conn.close()

# ─── equity helpers ───────────────────────────────────────────────────────────
def get_current_equity() -> float:
    """
    Return current account equity in USDT.
    Stubbed to read from env var or default to 1000.
    """
    return float(os.getenv("VIB_BOT_STARTING_EQUITY", "1000"))


def get_equity_curve() -> List[float]:
    """
    Return a simple equity curve for drawdown calculations.
    Stubbed to a single-point curve at current equity.
    """
    return [get_current_equity()]