#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# File paths – adjust if needed.
TRADES_FILE = "vib_trades_log.csv"
EXTRAS_FILE = "vib_extra_data.csv"
ORDERBOOK_FILE = "orderbook_data.csv"
OUTPUT_MERGED_FILE = "merged_training_data.csv"
MODEL_PATH = "model.pkl"

# Constants for training
BIG_TRADE_THRESHOLD = 100000       # Defines a "big" trade.
# Define thresholds for multi-class labels:
def label_candle(pct_change):
    if pct_change >= 0.10:
        return 3
    elif pct_change >= 0.05:
        return 2
    elif pct_change >= 0.01:
        return 1
    elif pct_change > -0.01:
        return 0
    elif pct_change > -0.05:
        return -1
    elif pct_change > -0.10:
        return -2
    else:
        return -3

def load_data():
    try:
        df_extras = pd.read_csv(EXTRAS_FILE, parse_dates=["open_time", "close_time"])
    except Exception as e:
        logger.error("Error loading extras CSV: %s", e)
        return None, None, None

    try:
        df_trades = pd.read_csv(TRADES_FILE, parse_dates=["LocalTime", "TradeTime"])
    except Exception as e:
        logger.error("Error loading trades CSV: %s", e)
        df_trades = pd.DataFrame()

    try:
        df_orderbook = pd.read_csv(ORDERBOOK_FILE)
        df_orderbook["timestamp"] = pd.to_datetime(df_orderbook["timestamp"], format="%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error("Error loading orderbook CSV: %s", e)
        df_orderbook = pd.DataFrame()

    df_extras.sort_values("open_time", inplace=True)
    return df_extras, df_trades, df_orderbook

def compute_big_trades_count(candle_close_time, trades_df):
    if trades_df.empty:
        return 0
    cutoff = candle_close_time - pd.Timedelta(minutes=5)
    recent_trades = trades_df[trades_df["TradeTime"] >= cutoff]
    big_trades = recent_trades[recent_trades["Quantity"] >= BIG_TRADE_THRESHOLD]
    return big_trades.shape[0]

def compute_orderbook_spread(candle_close_time, orderbook_df):
    if orderbook_df.empty:
        return 0.0
    valid_snapshots = orderbook_df[orderbook_df["timestamp"] <= candle_close_time]
    if valid_snapshots.empty:
        return 0.0
    latest_snapshot = valid_snapshots.iloc[-1]
    return latest_snapshot["spread"]

def create_training_dataset(df_extras, df_trades, df_orderbook):
    records = []
    for i in range(len(df_extras) - 1):
        current_row = df_extras.iloc[i]
        next_row = df_extras.iloc[i+1]
        
        # Extract VIB candle features.
        rsi = current_row.get("rsi", np.nan)
        macd_hist = current_row.get("macd_hist", np.nan)
        close_price = current_row.get("close", np.nan)
        volume = current_row.get("volume", np.nan)
        
        candle_close_time = current_row["close_time"]
        big_trades_count = compute_big_trades_count(candle_close_time, df_trades)
        orderbook_spread = compute_orderbook_spread(candle_close_time, df_orderbook)
        
        # Get correlation features from other symbols.
        btc_close = current_row.get("BTCUSDT", np.nan)
        eth_close = current_row.get("ETHUSDT", np.nan)
        rndr_close = current_row.get("RENDERUSDT", np.nan)
        diff_BTC = round((btc_close - close_price) / close_price, 0) if pd.notna(btc_close) and close_price != 0 else 0.0
        diff_ETH = round((eth_close - close_price) / close_price, 0) if pd.notna(eth_close) and close_price != 0 else 0.0
        diff_RNDR = round((rndr_close - close_price) / close_price, 0) if pd.notna(rndr_close) and close_price != 0 else 0.0
        
        try:
            pct_change = (next_row["close"] - close_price) / close_price
        except Exception:
            pct_change = 0.0
        label = label_candle(pct_change)
        
        records.append({
            "rsi": rsi,
            "macd_hist": macd_hist,
            "vib_close": close_price,
            "volume": volume,
            "big_trades_count": big_trades_count,
            "orderbook_spread": orderbook_spread,
            "diff_BTC": diff_BTC,
            "diff_ETH": diff_ETH,
            "diff_RNDR": diff_RNDR,
            "label": label
        })
    
    df_train = pd.DataFrame(records)
    df_train.dropna(inplace=True)
    return df_train

def main():
    df_extras, df_trades, df_orderbook = load_data()
    if df_extras is None:
        logger.error("Failed to load candle data. Exiting.")
        return

    logger.info("Loaded %d candle rows, %d trade rows, and %d orderbook snapshots.",
                len(df_extras), len(df_trades), len(df_orderbook))

    df_train = create_training_dataset(df_extras, df_trades, df_orderbook)
    if df_train.empty:
        logger.error("Training dataset is empty. Exiting.")
        return
    logger.info("Training dataset created with %d rows.", len(df_train))
    
    df_train.to_csv(OUTPUT_MERGED_FILE, index=False)
    logger.info("Merged training data saved to %s", OUTPUT_MERGED_FILE)

    # Use all 9 features.
    features = ["rsi", "macd_hist", "vib_close", "volume", "big_trades_count", 
                "orderbook_spread", "diff_BTC", "diff_ETH", "diff_RNDR"]
    X = df_train[features].values
    y = df_train["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Test Accuracy: %.2f", acc)
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    
    joblib.dump(clf, MODEL_PATH)
    logger.info("Model saved as %s", MODEL_PATH)

if __name__ == "__main__":
    main()