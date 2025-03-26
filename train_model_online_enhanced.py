#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
import os
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Define the base directory for your project
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

# Updated file paths using the base directory
TRADES_FILE = f"{BASE_DIR}/vib_trades_log.csv"
EXTRAS_FILE = f"{BASE_DIR}/vib_extra_data.csv"
ORDERBOOK_FILE = f"{BASE_DIR}/orderbook_data.csv"
OUTPUT_MERGED_FILE = f"{BASE_DIR}/merged_training_data.csv"
MODEL_PATH = f"{BASE_DIR}/model_online_enhanced.pkl"
SCALER_PATH = f"{BASE_DIR}/scaler_online.pkl"

# Setup logging (prints to console)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Constants for training
BIG_TRADE_THRESHOLD = 100000  # Defines a "big" trade.
LOOKAHEAD_SECONDS = 3600      # Lookahead window of 1 hour

def label_candle(pct_change):
    """
    Assigns a multi-class label based on the percentage change.
    """
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

def create_training_dataset_with_timeframe(df_extras, df_trades, df_orderbook, lookahead_seconds=LOOKAHEAD_SECONDS):
    records = []
    # Use only VIBUSDT data for training.
    df_vib = df_extras[df_extras["symbol"] == "VIBUSDT"].copy()
    df_vib.sort_values("close_time", inplace=True)
    
    # For correlation features, pivot the other symbols.
    other_symbols = ["BTCUSDT", "ETHUSDT", "RENDERUSDT"]
    df_others = df_extras[df_extras["symbol"].isin(other_symbols)].copy()
    if df_others.empty:
        logger.error("No data for other symbols.")
        return pd.DataFrame()
    df_others_pivot = df_others.pivot_table(index="close_time", columns="symbol", values="close", aggfunc="last")
    df_others_pivot.reset_index(inplace=True)
    
    # Merge VIB data with other symbols.
    df_merged = pd.merge_asof(df_vib, df_others_pivot, on="close_time", direction="backward", tolerance=pd.Timedelta(seconds=60))
    
    for i in range(len(df_merged)):
        current_row = df_merged.iloc[i]
        current_time = current_row["close_time"]
        
        # Select future rows within the lookahead timeframe.
        future_rows = df_merged[df_merged["close_time"] <= current_time + pd.Timedelta(seconds=lookahead_seconds)]
        if future_rows.empty:
            continue  # Skip if no future data
        
        # Compute future high and future low.
        future_high = future_rows["close"].max()
        future_low = future_rows["close"].min()
        
        # Use the more extreme move (up or down) relative to the current VIB close.
        vib_close = current_row.get("close", np.nan)
        if pd.isna(vib_close) or vib_close == 0:
            continue
        pct_change_up = (future_high - vib_close) / vib_close
        pct_change_down = (future_low - vib_close) / vib_close  # Negative if drop
        pct_change = pct_change_up if abs(pct_change_up) >= abs(pct_change_down) else pct_change_down
        label = label_candle(pct_change)
        
        # Extract additional features.
        rsi = current_row.get("rsi", np.nan)
        macd_hist = current_row.get("macd_hist", np.nan)
        volume = current_row.get("volume", np.nan)
        if volume == 0:
            continue  # Exclude rows with zero volume
        
        big_trades_count = compute_big_trades_count(current_time, df_trades)
        orderbook_spread = compute_orderbook_spread(current_time, df_orderbook)
        
        btc_close = current_row.get("BTCUSDT", np.nan)
        eth_close = current_row.get("ETHUSDT", np.nan)
        rndr_close = current_row.get("RENDERUSDT", np.nan)
        diff_BTC = round((btc_close - vib_close) / vib_close, 0) if pd.notna(btc_close) else 0.0
        diff_ETH = round((eth_close - vib_close) / vib_close, 0) if pd.notna(eth_close) else 0.0
        diff_RNDR = round((rndr_close - vib_close) / vib_close, 0) if pd.notna(rndr_close) else 0.0
        
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

def initial_training():
    df_extras, df_trades, df_orderbook = load_data()
    if df_extras is None or df_extras.empty:
        logger.error("No candle data available for training.")
        return None, None
    df_train = create_training_dataset_with_timeframe(df_extras, df_trades, df_orderbook)
    if df_train.empty:
        logger.error("Training dataset is empty.")
        return None, None
    logger.info("Training dataset created with %d rows.", len(df_train))
    
    # Save merged training data for inspection.
    df_train.to_csv(OUTPUT_MERGED_FILE, index=False)
    logger.info("Merged training data saved to %s", OUTPUT_MERGED_FILE)
    
    features = ["rsi", "macd_hist", "vib_close", "volume", "big_trades_count",
                "orderbook_spread", "diff_BTC", "diff_ETH", "diff_RNDR"]
    X = df_train[features].values
    y = df_train["label"].values
    
    # Feature scaling.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaler saved as %s", SCALER_PATH)
    
    # Hyperparameter tuning with GridSearchCV.
    from sklearn.linear_model import SGDClassifier
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01],
        "penalty": ["l2", "l1", "elasticnet"],
    }
    grid_search = GridSearchCV(SGDClassifier(loss="hinge", random_state=42, max_iter=1000, tol=1e-3),
                               param_grid, cv=3)
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    logger.info("Best parameters from GridSearchCV: %s", grid_search.best_params_)
    
    # Evaluate the best model.
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Initial Model Test Accuracy: %.2f", acc)
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Initial model saved as %s", MODEL_PATH)
    
    return best_model, scaler

def update_model_online(model, scaler, new_X, new_y):
    new_X_scaled = scaler.transform(new_X)
    model.partial_fit(new_X_scaled, new_y)
    joblib.dump(model, MODEL_PATH)
    logger.info("Online model update complete and model saved.")

def full_retrain_model(scaler):
    model, scaler = initial_training()
    return model, scaler

def main():
    # Initial training.
    model, scaler = initial_training()
    if model is None:
        logger.error("Initial training failed. Exiting.")
        return

    online_update_interval = 30   # seconds for online update.
    full_retrain_interval = 3600    # seconds (1 hour) for full retraining.
    last_full_retrain = time.time()

    while True:
        current_time = time.time()
        df_extras, df_trades, df_orderbook = load_data()
        new_data = create_training_dataset_with_timeframe(df_extras, df_trades, df_orderbook)
        if not new_data.empty:
            features = ["rsi", "macd_hist", "vib_close", "volume", "big_trades_count",
                        "orderbook_spread", "diff_BTC", "diff_ETH", "diff_RNDR"]
            new_X = new_data[features].values
            new_y = new_data["label"].values
            update_model_online(model, scaler, new_X, new_y)
            logger.info("Online model update complete.")
        else:
            logger.info("No new data for online update.")
        
        if current_time - last_full_retrain >= full_retrain_interval:
            model, scaler = full_retrain_model(scaler)
            last_full_retrain = current_time
            logger.info("Full retraining complete.")
        
        time.sleep(online_update_interval)

if __name__ == "__main__":
    main()