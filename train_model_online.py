#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
import os
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
MODEL_PATH = "model_online.pkl"  # New model file for online learning

# Constants for training
BIG_TRADE_THRESHOLD = 100000       # Defines a "big" trade.
# Multi-class labels: -3, -2, -1, 0, 1, 2, 3.
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
    # Use only VIBUSDT data for training; assume vib_extra_data has a 'symbol' column.
    df_vib = df_extras[df_extras["symbol"] == "VIBUSDT"].copy()
    df_vib.sort_values("close_time", inplace=True)
    
    # For correlation features, pivot the others.
    other_symbols = ["BTCUSDT", "ETHUSDT", "RENDERUSDT"]
    df_others = df_extras[df_extras["symbol"].isin(other_symbols)].copy()
    if df_others.empty:
        logger.error("No data for other symbols.")
        return pd.DataFrame()
    df_others_pivot = df_others.pivot_table(index="close_time", columns="symbol", values="close", aggfunc="last")
    df_others_pivot.reset_index(inplace=True)
    
    # Merge VIB data with other symbols' data
    df_merged = pd.merge_asof(df_vib, df_others_pivot, on="close_time", direction="backward", tolerance=pd.Timedelta(seconds=30))
    
    for i in range(len(df_merged) - 1):
        current_row = df_merged.iloc[i]
        next_row = df_merged.iloc[i+1]
        
        # Extract features
        rsi = current_row.get("rsi", np.nan)
        macd_hist = current_row.get("macd_hist", np.nan)
        vib_close = current_row.get("close", np.nan)
        volume = current_row.get("volume", np.nan)
        candle_close_time = current_row["close_time"]
        big_trades_count = compute_big_trades_count(candle_close_time, df_trades)
        orderbook_spread = compute_orderbook_spread(candle_close_time, df_orderbook)
        
        btc_close = current_row.get("BTCUSDT", np.nan)
        eth_close = current_row.get("ETHUSDT", np.nan)
        rndr_close = current_row.get("RENDERUSDT", np.nan)
        diff_BTC = round((btc_close - vib_close) / vib_close, 0) if pd.notna(btc_close) and vib_close != 0 else 0.0
        diff_ETH = round((eth_close - vib_close) / vib_close, 0) if pd.notna(eth_close) and vib_close != 0 else 0.0
        diff_RNDR = round((rndr_close - vib_close) / vib_close, 0) if pd.notna(rndr_close) and vib_close != 0 else 0.0
        
        try:
            pct_change = (next_row["close"] - vib_close) / vib_close
        except Exception:
            pct_change = 0.0
        label = label_candle(pct_change)
        
        records.append({
            "rsi": rsi,
            "macd_hist": macd_hist,
            "vib_close": vib_close,
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

def train_initial_model():
    df_extras, df_trades, df_orderbook = load_data()
    if df_extras is None or df_extras.empty:
        logger.error("No candle data available for training.")
        return None, None, None
    df_train = create_training_dataset(df_extras, df_trades, df_orderbook)
    if df_train.empty:
        logger.error("Training dataset is empty.")
        return None, None, None
    logger.info("Training dataset created with %d rows.", len(df_train))
    
    # Save merged training data for inspection.
    df_train.to_csv(OUTPUT_MERGED_FILE, index=False)
    logger.info("Merged training data saved to %s", OUTPUT_MERGED_FILE)
    
    # Define features (9 features).
    features = ["rsi", "macd_hist", "vib_close", "volume", "big_trades_count", 
                "orderbook_spread", "diff_BTC", "diff_ETH", "diff_RNDR"]
    X = df_train[features].values
    y = df_train["label"].values
    # Use classes from -3 to 3.
    classes = np.array([-3, -2, -1, 0, 1, 2, 3])
    
    # Split into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use an SGDClassifier for online learning.
    from sklearn.linear_model import SGDClassifier
    model = SGDClassifier(loss="hinge", random_state=42, max_iter=1000, tol=1e-3)
    # For initial training, use partial_fit with the full training data.
    model.partial_fit(X_train, y_train, classes=classes)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Initial Model Test Accuracy: %.2f", acc)
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    
    # Save the model.
    joblib.dump(model, MODEL_PATH)
    logger.info("Initial model saved as %s", MODEL_PATH)
    
    return model, X_train, y_train

def update_model_online(model, new_X, new_y):
    # Update the model using partial_fit on new data.
    model.partial_fit(new_X, new_y)
    # Optionally, save the updated model.
    joblib.dump(model, MODEL_PATH)
    logger.info("Model updated online and saved.")

def full_retrain_model():
    # Full retraining from scratch using all available data.
    model, _, _ = train_initial_model()
    return model

def main():
    # Initial training
    model, X_train, y_train = train_initial_model()
    if model is None:
        logger.error("Initial training failed. Exiting.")
        return

    # We'll simulate online updates and full retraining.
    online_update_interval = 30  # seconds
    full_retrain_interval = 3600  # seconds (1 hour)
    last_full_retrain = time.time()
    
    while True:
        current_time = time.time()
        
        # For online update, load new data (for demonstration we use all data)
        df_extras, df_trades, df_orderbook = load_data()
        new_data = create_training_dataset(df_extras, df_trades, df_orderbook)
        if not new_data.empty:
            features = ["rsi", "macd_hist", "vib_close", "volume", "big_trades_count", 
                        "orderbook_spread", "diff_BTC", "diff_ETH", "diff_RNDR"]
            new_X = new_data[features].values
            new_y = new_data["label"].values
            update_model_online(model, new_X, new_y)
            logger.info("Online model update complete.")
        else:
            logger.info("No new data for online update.")
        
        # Check if it's time for full retraining.
        if current_time - last_full_retrain >= full_retrain_interval:
            model = full_retrain_model()
            last_full_retrain = current_time
            logger.info("Full retraining complete.")
        
        time.sleep(online_update_interval)

if __name__ == "__main__":
    main()