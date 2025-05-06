#!/usr/bin/env python3
"""
processing/utils.py

Helpers for loading raw data and computing feature windows
for training‑data generation.
"""
import sqlite3
from datetime import timedelta

import pandas as pd
import numpy as np

from vib_bot.config import (
    EXTRAS_DB_FILE,
    TRADES_DB_FILE,
    ORDERBOOK_DB_FILE,
    BIG_TRADE_THRESHOLD,
    CORRELATION_SYMBOLS,
)


def load_extras() -> pd.DataFrame:
    """
    Load indicator (RSI, MACD, etc.) data for all symbols.
    """
    with sqlite3.connect(EXTRAS_DB_FILE) as conn:
        df = pd.read_sql_query(
            "SELECT symbol, open_time, close_time, close, volume, rsi, macd_hist "
            "FROM vib_extra_data",
            conn,
            parse_dates=["open_time", "close_time"]
        )
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df


def load_trades() -> pd.DataFrame:
    """
    Load raw trade data (timestamp + volume).
    """
    with sqlite3.connect(TRADES_DB_FILE) as conn:
        df = pd.read_sql_query(
            "SELECT trade_time, quantity FROM trades",
            conn,
            parse_dates=["trade_time"]
        )
    df["trade_time"] = pd.to_datetime(df["trade_time"], utc=True)
    return df


def load_orderbook() -> pd.DataFrame:
    """
    Load orderbook snapshots (timestamp + spread).
    """
    with sqlite3.connect(ORDERBOOK_DB_FILE) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, spread FROM orderbook_data",
            conn,
            parse_dates=["timestamp"]
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def bucket_label(pct: float) -> int:
    """
    Map a percent change into one of seven buckets: -3..+3.
    """
    if pct >= 0.10:
        return 3
    elif pct >= 0.05:
        return 2
    elif pct >= 0.01:
        return 1
    elif pct > -0.01:
        return 0
    elif pct > -0.05:
        return -1
    elif pct > -0.10:
        return -2
    else:
        return -3


def compute_feature_windows(
    extras: pd.DataFrame,
    trades: pd.DataFrame,
    orderbook: pd.DataFrame,
    row,
    look_ahead: int
) -> tuple:
    """
    For a single `row` (one asset’s candle), compute:
      1) big_trades_count (qty ≥ BIG_TRADE_THRESHOLD)
      2) latest orderbook spread
      3) relative growth diffs for each symbol in CORRELATION_SYMBOLS _other_ than row.symbol

    Returns: (big_trades_count, spread, diff_1, diff_2, …)
    """
    t0 = row.close_time - timedelta(minutes=look_ahead)
    t1 = row.close_time
    base_close = row.close
    vib_pct = (row.future_close - base_close) / base_close

    # 1) Big‐trade count in the look‐ahead window
    mask = (
        (trades.trade_time >= t0)
        & (trades.trade_time <= t1)
        & (trades.quantity   >= BIG_TRADE_THRESHOLD)
    )
    big_count = int(mask.sum())

    # 2) Latest spread
    ob = orderbook[orderbook.timestamp <= t1]
    spread = ob.spread.iloc[-1] if not ob.empty else np.nan

    # 3) Relative diffs for each correlation symbol
    diffs = []
    for sym in CORRELATION_SYMBOLS:
        if sym == row.symbol:
            # skip comparing the asset to itself
            continue
        window = extras[
            (extras.symbol     == sym) &
            (extras.close_time >= t0)    &
            (extras.close_time <= t1)
        ].sort_values("close_time")

        if len(window) >= 2:
            pct = (window.close.iloc[-1] - window.close.iloc[0]) / window.close.iloc[0]
            diffs.append(1.0 + (pct - vib_pct))
        else:
            diffs.append(1.0)

    return (big_count, spread, *diffs)