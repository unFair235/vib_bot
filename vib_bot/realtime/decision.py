#!/usr/bin/env python3
"""
realtime/decision.py

Macro‐filters & map features+label → LONG/SHORT/HOLD
"""
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from vib_bot.config import EXTRAS_DB_FILE, BIG_TRADE_THRESHOLD

logger = logging.getLogger("decision")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


def _btc_up_too_much(threshold: float = 0.03, lookback_hours: float = 4.0) -> bool:
    """
    Returns True if BTCUSDT has risen more than `threshold` in the past `lookback_hours`.
    """
    since = datetime.utcnow() - timedelta(hours=lookback_hours)
    query = """
      SELECT close_time, close
      FROM vib_extra_data
      WHERE symbol='BTCUSDT'
        AND close_time >= ?
      ORDER BY close_time ASC
    """
    with sqlite3.connect(EXTRAS_DB_FILE) as conn:
        df = pd.read_sql_query(query, conn, params=(since.isoformat(),), parse_dates=["close_time"])
    if len(df) < 2:
        return False
    start, end = df["close"].iloc[0], df["close"].iloc[-1]
    rise = (end - start) / start if start else 0.0
    return rise > threshold


def _no_major_news() -> bool:
    """
    Stub for a news‐based filter. Return False (i.e. OK to trade).
    """
    return True


def make_decision(
    features: np.ndarray,
    label: int,
    equity: float = None
) -> Optional[str]:
    """
    Given your 1×9 feature vector and model label in [-3..3], return:
      - "LONG"   (for a buy)
      - "SHORT"  (for a short)
      - None     (no trade)
    Applies macro‐filters first, then simple label thresholds.
    """
    # 1) Macro Filters
    if _btc_up_too_much():
        logger.info("Skipping trade: BTC up >3% in last 4h")
        return None
    if not _no_major_news():
        logger.info("Skipping trade: major news flag")
        return None

    # 2) Label‐based trigger
    if label >= 2:
        action = "LONG"
    elif label <= -2:
        action = "SHORT"
    else:
        action = None

    logger.info(f"Decision: label={label} → action={action}")
    return action