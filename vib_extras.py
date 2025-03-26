#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import time
import os
import sqlite3
import ta  # Technical Analysis library

# ----------------------------
# Configuration & File Paths
# ----------------------------
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"
DB_FILE = os.path.join(BASE_DIR, "vib_extra_data.db")  # Database for extra data

# Binance API configuration
API_URL = "https://api.binance.com/api/v3/klines"
KLINE_INTERVAL = "1m"
KLINE_LIMIT = 500

# List of symbols to fetch
SYMBOLS = ["VIBUSDT", "BTCUSDT", "ETHUSDT", "RENDERUSDT"]

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("vib_extras")

# ----------------------------
# Database Functions
# ----------------------------
def init_extra_data_db():
    """Initialize the SQLite database for extra data and create the vib_extra_data table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vib_extra_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            open_time TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            close_time TEXT,
            quote_asset_volume REAL,
            number_of_trades INTEGER,
            taker_buy_base REAL,
            taker_buy_quote REAL,
            ignore TEXT,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            symbol TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized (vib_extra_data table ensured).")

def store_extra_data(df):
    """Store the merged extra data DataFrame into the vib_extra_data table."""
    conn = sqlite3.connect(DB_FILE)
    # Write the DataFrame into the table; if the table exists, replace its contents
    df.to_sql("vib_extra_data", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    logger.info(f"Stored extra data into database with {len(df)} rows.")

# Initialize the database at startup
init_extra_data_db()

# ----------------------------
# Helper Functions for Data Fetching & Indicators
# ----------------------------
def fetch_klines(symbol, interval, limit=KLINE_LIMIT):
    """Fetch klines for a given symbol from Binance."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        data = response.json()
        # If an error occurs, data might be a dict with an error code.
        if isinstance(data, dict) and data.get("code"):
            logger.error(f"Error fetching klines for {symbol}: {data}")
            return pd.DataFrame()
        # Define column names per Binance API documentation.
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(data, columns=columns)
        # Convert timestamps from ms to datetime
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        # Convert numeric columns to floats
        for col in ["open", "high", "low", "close", "volume", 
                    "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["symbol"] = symbol
        return df
    except Exception as e:
        logger.error(f"Exception fetching klines for {symbol}: {e}")
        return pd.DataFrame()

def compute_indicators(df):
    """Compute RSI and MACD for the DataFrame; fill NaN values with 0."""
    try:
        df["rsi"] = ta.momentum.rsi(df["close"], window=14).fillna(0)
    except Exception as e:
        logger.error(f"Error computing RSI: {e}")
        df["rsi"] = 0
    try:
        macd = ta.trend.macd(df["close"])
        macd_signal = ta.trend.macd_signal(df["close"])
        df["macd"] = macd.fillna(0)
        df["macd_signal"] = macd_signal.fillna(0)
        df["macd_hist"] = (macd - macd_signal).fillna(0)
    except Exception as e:
        logger.error(f"Error computing MACD: {e}")
        df["macd"] = 0
        df["macd_signal"] = 0
        df["macd_hist"] = 0
    return df

# ----------------------------
# Main Function
# ----------------------------
def main():
    all_data = []
    for symbol in SYMBOLS:
        logger.info(f"Fetching klines for {symbol}...")
        df = fetch_klines(symbol, KLINE_INTERVAL, limit=KLINE_LIMIT)
        if df.empty:
            logger.warning(f"No data fetched for {symbol}.")
            continue
        df = compute_indicators(df)
        all_data.append(df)
        time.sleep(0.2)  # Delay to avoid hitting API rate limits

    if not all_data:
        logger.error("No data fetched for any symbol.")
        return

    # Concatenate data for all symbols
    df_all = pd.concat(all_data, ignore_index=True)
    # Sort data by symbol and close_time
    df_all.sort_values(by=["symbol", "close_time"], inplace=True)
    logger.info(f"Fetched and computed indicators for {len(df_all)} rows across all symbols.")

    # Instead of writing to CSV, store the combined DataFrame into SQLite
    store_extra_data(df_all)
    logger.info("Extra data stored in the SQLite database.")

if __name__ == "__main__":
    main()