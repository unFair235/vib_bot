#!/usr/bin/env python3
import time
import logging
import sqlite3
import requests
import pandas as pd

from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

import ta  # Technical Analysis library

from vib_bot.config import EXTRAS_DB_FILE, SYMBOLS, DATA_FRESHNESS_THRESHOLD

# ────────────────────────────────────────────────────────────────────────────────
# Binance Klines configuration
# ────────────────────────────────────────────────────────────────────────────────
API_URL       = "https://api.binance.com/api/v3/klines"
KLINE_INTERVAL = "1m"
KLINE_LIMIT    = 500

DB_FILE = EXTRAS_DB_FILE

# ────────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ────────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("vib_extras")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# console
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

# rotating file
log_path = DB_FILE.replace(".db", "_extras.log")
fh = RotatingFileHandler(log_path, maxBytes=5 * 1024**2, backupCount=3)
fh.setFormatter(fmt)
logger.addHandler(fh)


# ────────────────────────────────────────────────────────────────────────────────
# Database Functions
# ────────────────────────────────────────────────────────────────────────────────
def init_extra_data_db():
    """Ensure vib_extra_data table exists with a `symbol` column."""
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vib_extra_data (
            open_time            TEXT,
            open                 REAL,
            high                 REAL,
            low                  REAL,
            close                REAL,
            volume               REAL,
            close_time           TEXT,
            quote_asset_volume   REAL,
            number_of_trades     INTEGER,
            taker_buy_base       REAL,
            taker_buy_quote      REAL,
            ignore               TEXT,
            rsi                  REAL,
            macd                 REAL,
            macd_signal          REAL,
            macd_hist            REAL,
            symbol               TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Initialized vib_extra_data table.")


def store_extra_data(df: pd.DataFrame):
    """Replace the vib_extra_data table with the latest concatenated DataFrame."""
    conn = sqlite3.connect(DB_FILE)
    df.to_sql("vib_extra_data", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    logger.info(f"Stored {len(df)} rows into vib_extra_data.")


# Initialize on import
init_extra_data_db()


# ────────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────────
def fetch_klines(symbol: str,
                 interval: str,
                 limit: int = KLINE_LIMIT,
                 retries: int = 3,
                 delay: int = 5) -> pd.DataFrame:
    """
    Fetch candlesticks for one symbol, retrying up to `retries`.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # error?
            if isinstance(data, dict) and data.get("code"):
                raise ValueError(data)
            # build DataFrame
            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]
            df = pd.DataFrame(data, columns=cols)
            df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            # cast numeric
            for c in ["open", "high", "low", "close", "volume",
                      "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["symbol"] = symbol
            return df
        except Exception as e:
            logger.error(f"[{symbol}] fetch attempt {attempt} failed: {e}")
            time.sleep(delay)
    logger.error(f"[{symbol}] all {retries} attempts failed, returning empty DataFrame.")
    return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, MACD, MACD signal, and MACD histogram.
    """
    # RSI
    try:
        df["rsi"] = ta.momentum.rsi(df["close"], window=14).fillna(0.0)
    except Exception:
        df["rsi"] = 0.0

    # MACD
    try:
        macd        = ta.trend.macd(df["close"])
        macd_signal = ta.trend.macd_signal(df["close"])
        df["macd"]       = macd.fillna(0.0)
        df["macd_signal"] = macd_signal.fillna(0.0)
        df["macd_hist"]  = (macd - macd_signal).fillna(0.0)
    except Exception:
        df["macd"]       = 0.0
        df["macd_signal"] = 0.0
        df["macd_hist"]  = 0.0

    return df


# ────────────────────────────────────────────────────────────────────────────────
# Main Loop
# ────────────────────────────────────────────────────────────────────────────────
def main():
    if not SYMBOLS:
        logger.error("No SYMBOLS configured; exiting.")
        return

    while True:
        all_frames = []
        for sym in SYMBOLS:
            logger.info(f"Fetching klines for {sym} …")
            df = fetch_klines(sym, KLINE_INTERVAL)
            if df.empty:
                logger.warning(f"No data for {sym}, skipping.")
                continue
            all_frames.append(compute_indicators(df))
            time.sleep(0.2)

        if not all_frames:
            logger.error("Failed to fetch data for any symbol; retrying.")
        else:
            concat_df = pd.concat(all_frames, ignore_index=True)
            concat_df.sort_values(["symbol", "close_time"], inplace=True)
            last_time = concat_df["open_time"].max()
            age = (datetime.now(timezone.utc) - last_time).total_seconds()
            if age > DATA_FRESHNESS_THRESHOLD:
                logger.error(f"Data is stale by {age:.0f}s; skipping store.")
            else:
                store_extra_data(concat_df)

        time.sleep(30)


if __name__ == "__main__":
    main()