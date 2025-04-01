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

# Import Keras load_model for neural network loading
from tensorflow.keras.models import load_model

# ----------------------------
# Configuration & File Paths
# ----------------------------
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

TRADES_DB_FILE = os.path.join(BASE_DIR, "trades.db")
EXTRAS_DB_FILE = os.path.join(BASE_DIR, "vib_extra_data.db")
ORDERBOOK_DB_FILE = os.path.join(BASE_DIR, "orderbook.db")

# Model paths for different models
MODEL_PATH_LINEAR = os.path.join(BASE_DIR, "model.pkl")       # Linear model
MODEL_PATH_NN = os.path.join(BASE_DIR, "model_nn.h5")           # Neural network model
SCALER_PATH_NN = os.path.join(BASE_DIR, "scaler_nn.pkl")        # Scaler for NN input
ACTIVE_MODEL_FILE = os.path.join(BASE_DIR, "active_model.txt")   # Contains "linear" or "nn"

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
# Schema Ensurer Function
# ----------------------------
def ensure_schema():
    """Ensure that the 'predictions' and 'pending_feedback' tables have the 'model_id' column."""
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    
    # Check predictions table
    cur.execute("PRAGMA table_info(predictions)")
    columns = [row[1] for row in cur.fetchall()]
    if "model_id" not in columns:
        try:
            cur.execute("ALTER TABLE predictions ADD COLUMN model_id TEXT")
            conn.commit()
            logger.info("Added 'model_id' column to predictions table.")
        except Exception as e:
            logger.error(f"Error adding 'model_id' column to predictions: {e}")
    
    # Check pending_feedback table
    cur.execute("PRAGMA table_info(pending_feedback)")
    columns = [row[1] for row in cur.fetchall()]
    if "model_id" not in columns:
        try:
            cur.execute("ALTER TABLE pending_feedback ADD COLUMN model_id TEXT")
            conn.commit()
            logger.info("Added 'model_id' column to pending_feedback table.")
        except Exception as e:
            logger.error(f"Error adding 'model_id' column to pending_feedback: {e}")
    conn.close()

# Call ensure_schema() at startup
ensure_schema()

# ----------------------------
# Active Model Loader
# ----------------------------
def load_active_model():
    """
    Loads the active model based on the configuration in active_model.txt.
    Returns a tuple ((model, scaler), model_id) where:
      - For the neural network ('nn'), scaler is loaded.
      - For the linear model, scaler is None.
    """
    try:
        with open(ACTIVE_MODEL_FILE, "r") as f:
            active = f.read().strip()
    except Exception as e:
        logger.error(f"Error reading active model config: {e}")
        active = "linear"  # default if file missing

    if active == "nn":
        try:
            model = load_model(MODEL_PATH_NN)
            scaler = joblib.load(SCALER_PATH_NN)
            logger.info("Active model set to neural network.")
            return (model, scaler), "nn"
        except Exception as e:
            logger.error(f"Error loading neural network model: {e}")
            # Fallback to linear model if error occurs
            model = joblib.load(MODEL_PATH_LINEAR)
            return (model, None), "linear"
    else:
        try:
            model = joblib.load(MODEL_PATH_LINEAR)
            logger.info("Active model set to linear.")
            return (model, None), "linear"
        except Exception as e:
            logger.error(f"Error loading linear model: {e}")
            return None, None

# ----------------------------
# DB Write Functions for Master DB
# ----------------------------
def store_prediction(timestamp, predicted_label, model_id):
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO predictions (timestamp, predicted_label, model_id) VALUES (?, ?, ?)",
                (timestamp, predicted_label, model_id))
    conn.commit()
    conn.close()

def store_pending_feedback(timestamp, predicted_label, features, vib_price, model_id):
    features_json = json.dumps(features.tolist())
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO pending_feedback (timestamp, predicted_label, features, model_id) VALUES (?, ?, ?, ?)",
                (timestamp, predicted_label, features_json, model_id))
    conn.commit()
    conn.close()

def store_alert(message):
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
def run_inference():
    # Load active model (linear or neural network) along with scaler if NN
    (model, scaler), model_id = load_active_model()
    if model is None:
        logger.error("No active model loaded. Aborting inference.")
        return

    df_extras, df_trades, df_orderbook = load_data()
    merged = merge_data(df_extras, df_trades, df_orderbook)
    if merged is None:
        logger.warning("[ML] Merge failed. Skipping inference cycle.")
        return
    X, current_vib_price, current_timestamp = merged

    # If using neural network, scale features using the saved scaler
    if model_id == "nn" and scaler is not None:
        X = scaler.transform(X)
    
    try:
        prediction = model.predict(X)[0]
    except Exception as e:
        logger.error(f"[ML] Prediction error: {e}")
        return
    
    timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    store_prediction(timestamp_str, prediction, model_id)
    logger.info(f"[ML] Model {model_id} Prediction: {prediction}")
    # Store pending feedback record with model_id
    store_pending_feedback(timestamp_str, prediction, X, current_vib_price, model_id)
    
    # Define a mapping for actionable signals
    action_map = {
        3: "Strong Buy",
        2: "Buy",
        1: "Slight Buy",
        0: "Hold",
        -1: "Slight Sell",
        -2: "Sell",
        -3: "Strong Sell"
    }
    action = action_map.get(prediction, "Unknown Signal")
    
    # Extract key features for context
    # Assuming feature order: [rsi, macd_hist, close, volume, big_trades_count, orderbook_spread, diff_BTC, diff_ETH, diff_RNDR]
    features_list = X[0]
    alert_msg = (
        f"Model {model_id} Signal: {action} (Prediction: {prediction})\n"
        f"VIB Price: {features_list[2]:.5f}\n"
        f"Volume: {features_list[3]:.1f}\n"
        f"BTC Corr: {features_list[6]:.2f}\n"
        f"ETH Corr: {features_list[7]:.2f}\n"
        f"RNDR Corr: {features_list[8]:.2f}"
    )
    if prediction != 0:
        send_telegram_alert(alert_msg)

def main_loop():
    while True:
        run_inference()
        time.sleep(30)

if __name__ == "__main__":
    main_loop()