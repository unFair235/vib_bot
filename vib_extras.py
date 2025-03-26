#!/usr/bin/env python3
import os
import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("vib_extra")

# Define the base directory for your project
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

# Binance API configuration
SYMBOLS = ["VIBUSDT", "ETHUSDT", "BTCUSDT", "RENDERUSDT"]
API_URL = "https://api.binance.com/api/v3/klines"
KLINE_LIMIT = 500  
KLINE_INTERVAL = "1m"

# CSV file to store combined data + indicators (absolute path)
CSV_FILE = os.path.join(BASE_DIR, "vib_extra_data.csv")

def get_klines(symbol, interval, limit=KLINE_LIMIT):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

    # Expected columns from Binance API
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=columns)

    # Convert numeric columns to float (using pd.to_numeric for safety)
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume",
                "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert time columns from milliseconds to datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df

def compute_indicators(df):
    if df.empty:
        return df
    # Use the close price for technical indicators.
    close_series = df["close"]
    df["rsi"] = ta.momentum.rsi(close_series, window=14)
    macd = ta.trend.macd(close_series)
    macd_signal = ta.trend.macd_signal(close_series)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal
    return df

def main():
    all_data = []
    for symbol in SYMBOLS:
        df_klines = get_klines(symbol, KLINE_INTERVAL)
        if df_klines.empty:
            continue
        df_klines = compute_indicators(df_klines)
        df_klines["symbol"] = symbol
        all_data.append(df_klines)
    
    if not all_data:
        logger.error("No data fetched for any symbol.")
        return

    # Concatenate data from all symbols.
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Sort data by symbol and close_time.
    df_all.sort_values(by=["symbol", "close_time"], inplace=True)
    
    # Append data to CSV; write header only if file doesn't exist.
    file_exists = os.path.exists(CSV_FILE)
    try:
        df_all.to_csv(CSV_FILE, mode='a', header=not file_exists, index=False)
        logger.info("Data fetched & appended to CSV. Last 5 rows:")
        logger.info(df_all.tail(5).to_string())
    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")
    
    # OPTIONAL: Compute correlation matrices.
    try:
        pivoted = df_all.pivot_table(index="close_time", columns="symbol", values="close", aggfunc="last")
        pivoted.reset_index(inplace=True)
        numeric_pivot = pivoted.select_dtypes(include=[np.number])
        logger.info("\nCorrelation Matrix of close prices:")
        logger.info(numeric_pivot.corr(method="pearson").to_string())
        
        returns = numeric_pivot.pct_change()
        logger.info("\nCorrelation Matrix of 1m returns:")
        logger.info(returns.corr().to_string())
    except Exception as e:
        logger.error(f"Error computing correlations: {e}")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(60)  # Wait 60 seconds before next fetch