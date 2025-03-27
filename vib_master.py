#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import time
import os
import logging
import sqlite3
import json
import requests

# ----------------------------
# Configuration & File Paths
# ----------------------------
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

TRADES_DB_FILE = os.path.join(BASE_DIR, "trades.db")
EXTRAS_DB_FILE = os.path.join(BASE_DIR, "vib_extra_data.db")
ORDERBOOK_DB_FILE = os.path.join(BASE_DIR, "orderbook.db")

# Model path (trained model to be used for inference)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Master DB for storing predictions and pending feedback (for training updates)
MASTER_DB_FILE = os.path.join(BASE_DIR, "vib_master.db")

TELEGRAM_TOKEN = "7636229600:AAESoUoIB6nIcUHxme43x8byKhX1sok5zPk"
CHAT_ID = 531265494

# Feedback window (in seconds)
FEEDBACK_WINDOW = 3600

# ----------------------------
# Logging Setup
# ----------------------------
logger = logging.getLogger("vib_master")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(BASE_DIR, "vib_master.log"))
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ----------------------------
# DB Write Functions for Master DB
# ----------------------------
def store_prediction(timestamp, predicted_label):
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO predictions (timestamp, predicted_label) VALUES (?, ?)",
                (timestamp, predicted_label))
    conn.commit()
    conn.close()

def store_pending_feedback(timestamp, predicted_label, features, vib_price):
    # Save features as JSON
    features_json = json.dumps(features.tolist())
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO pending_feedback (timestamp, predicted_label, features) VALUES (?, ?, ?)",
                (timestamp, predicted_label, features_json))
    conn.commit()
    conn.close()

def store_alert(message):
    # Optional: log alert messages or perform further actions.
    logger.info(f"Telegram Alert Sent: {message}")

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        resp = requests.post(url, data=data, timeout=5)
        if resp.status_code != 200:
            logger.error(f"Telegram Error: {resp.text}")
    except Exception as e:
        logger.error(f"Telegram Exception: {e}")

# ----------------------------
# Data Loading Functions
# ----------------------------
def load_trades_data():
    try:
        conn = sqlite3.connect(TRADES_DB_FILE)
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()
        if "trade_time" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
        if "local_time" in df.columns:
            df["local_time"] = pd.to_datetime(df["local_time"], errors="coerce")
        logger.info(f"Loaded trades data: {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error loading trades data: {e}")
        df = pd.DataFrame()
    return df

def load_extras_data():
    try:
        conn = sqlite3.connect(EXTRAS_DB_FILE)
        df = pd.read_sql_query("SELECT * FROM vib_extra_data", conn, parse_dates=["open_time", "close_time"])
        conn.close()
        logger.info(f"Loaded extras data: {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error loading extras data: {e}")
        df = pd.DataFrame()
    return df

def load_orderbook_data():
    try:
        conn = sqlite3.connect(ORDERBOOK_DB_FILE)
        df = pd.read_sql_query("SELECT * FROM orderbook_data", conn, parse_dates=["timestamp"])
        conn.close()
        df.sort_values("timestamp", inplace=True)
        logger.info(f"Loaded orderbook data: {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error loading orderbook data: {e}")
        df = pd.DataFrame()
    return df

def load_data():
    df_extras = load_extras_data()
    df_trades = load_trades_data()
    df_orderbook = load_orderbook_data()
    if not df_extras.empty:
        df_extras.sort_values("open_time", inplace=True)
    return df_extras, df_trades, df_orderbook

# ----------------------------
# Data Merging Function
# ----------------------------
def merge_data(df_extras, df_trades, df_orderbook):
    # Filter VIBUSDT data
    df_vib = df_extras[df_extras["symbol"] == "VIBUSDT"].copy()
    if df_vib.empty:
        logger.error("No VIBUSDT data found.")
        return None
    df_vib.sort_values("close_time", inplace=True)
    latest_candle = df_vib.iloc[-1]
    # Count big trades (last 5 minutes)
    BIG_TRADE_THRESHOLD = 100000
    big_trades_count = len(df_trades[
        (df_trades["trade_time"] >= latest_candle["close_time"] - timedelta(minutes=5)) &
        (df_trades["quantity"] >= BIG_TRADE_THRESHOLD)
    ])
    # Get orderbook spread
    if not df_orderbook.empty:
        valid_snapshots = df_orderbook[df_orderbook["timestamp"] <= latest_candle["close_time"]]
        orderbook_spread = valid_snapshots.iloc[-1]["spread"] if not valid_snapshots.empty else 0.0
    else:
        orderbook_spread = 0.0
    # Pivot table for related symbols
    other_symbols = ["BTCUSDT", "ETHUSDT", "RENDERUSDT"]
    df_others = df_extras[df_extras["symbol"].isin(other_symbols)].copy()
    if df_others.empty:
        diff_BTC, diff_ETH, diff_RNDR = 0, 0, 0
    else:
        df_pivot = df_others.pivot_table(index="close_time", columns="symbol", values="close", aggfunc="last")
        df_pivot.reset_index(inplace=True)
        df_merged = pd.merge_asof(df_vib, df_pivot, on="close_time", direction="backward", tolerance=pd.Timedelta(seconds=60))
        diff_BTC = (df_merged.iloc[-1].get("BTCUSDT", np.nan) - latest_candle.get("close", 0)) / latest_candle.get("close", 1)
        diff_ETH = (df_merged.iloc[-1].get("ETHUSDT", np.nan) - latest_candle.get("close", 0)) / latest_candle.get("close", 1)
        diff_RNDR = (df_merged.iloc[-1].get("RENDERUSDT", np.nan) - latest_candle.get("close", 0)) / latest_candle.get("close", 1)
        diff_BTC = 0 if np.isnan(diff_BTC) else diff_BTC
        diff_ETH = 0 if np.isnan(diff_ETH) else diff_ETH
        diff_RNDR = 0 if np.isnan(diff_RNDR) else diff_RNDR
    features = [
        latest_candle.get("rsi", 0),
        latest_candle.get("macd_hist", 0),
        latest_candle.get("close", 0),
        latest_candle.get("volume", 0),
        big_trades_count,
        orderbook_spread,
        diff_BTC,
        diff_ETH,
        diff_RNDR
    ]
    # Return: features (numpy array), current VIB price, and the candle's timestamp
    return np.array([features]), latest_candle["close"], latest_candle["close_time"]

# ----------------------------
# Inference Loop
# ----------------------------
def run_inference(model):
    df_extras, df_trades, df_orderbook = load_data()
    merged = merge_data(df_extras, df_trades, df_orderbook)
    if merged is None:
        logger.warning("[ML] Merge failed. Skipping inference cycle.")
        return
    X, current_vib_price, current_timestamp = merged
    try:
        prediction = model.predict(X)[0]
    except Exception as e:
        logger.error(f"[ML] Prediction error: {e}")
        return
    timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    store_prediction(timestamp_str, prediction)
    logger.info(f"[ML] Prediction: {prediction}")
    # Store pending feedback record for later training update
    store_pending_feedback(timestamp_str, prediction, X, current_vib_price)
    if prediction != 0:
        alert_msg = f"Signal: {prediction}\nFeatures: {X[0].tolist()}"
        send_telegram_alert(alert_msg)

def main_loop():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}.")
        return
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully for inference.")
    while True:
        run_inference(model)
        time.sleep(30)

if __name__ == "__main__":
    main_loop()