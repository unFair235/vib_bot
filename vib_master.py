#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import time
import os
import logging
import requests
import sqlite3

# ----------------------------
# Configuration & File Paths
# ----------------------------

# Set the base directory for your project
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

# CSV files (for trades, extras, orderbook)
TRADES_FILE = os.path.join(BASE_DIR, "vib_trades_log.csv")
EXTRAS_FILE = os.path.join(BASE_DIR, "vib_extra_data.csv")
ORDERBOOK_FILE = os.path.join(BASE_DIR, "orderbook_data.csv")

# Model file path
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# SQLite DB file to store predictions and feedback
DB_FILE = os.path.join(BASE_DIR, "vib_master.db")

# Telegram Bot Configuration
TELEGRAM_TOKEN = "7636229600:AAESoUoIB6nIcUHxme43x8byKhX1sok5zPk"
CHAT_ID = 531265494

# Feedback window (in seconds)
FEEDBACK_WINDOW = 3600  # 1 hour

# ----------------------------
# Logging Setup
# ----------------------------

logger = logging.getLogger("vib_master")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(BASE_DIR, "vib_master.log"))
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# ----------------------------
# Global Variables & Caching
# ----------------------------

pending_predictions = []  # List to hold predictions pending feedback
_cached_data = None
_last_mod_times = {}

def get_file_mod_time(filepath):
    try:
        return os.path.getmtime(filepath)
    except Exception as e:
        logger.error(f"Error getting mod time for {filepath}: {e}")
        return None

def load_data():
    """
    Loads CSV files for trades, extras, and orderbook.
    Caches the DataFrames if files haven't changed.
    """
    global _cached_data, _last_mod_times
    mod_times = {
        "trades": get_file_mod_time(TRADES_FILE),
        "extras": get_file_mod_time(EXTRAS_FILE),
        "orderbook": get_file_mod_time(ORDERBOOK_FILE)
    }
    if _cached_data is not None and mod_times == _last_mod_times:
        return _cached_data

    try:
        df_trades = pd.read_csv(TRADES_FILE, parse_dates=["LocalTime", "TradeTime"])
        logger.info(f"Loaded trades data: {len(df_trades)} rows.")
    except Exception as e:
        logger.error(f"Error loading trades CSV: {e}")
        df_trades = pd.DataFrame()
    try:
        df_extras = pd.read_csv(EXTRAS_FILE, parse_dates=["open_time", "close_time"])
        logger.info(f"Loaded extras data: {len(df_extras)} rows.")
    except Exception as e:
        logger.error(f"Error loading extras CSV: {e}")
        df_extras = pd.DataFrame()
    try:
        df_orderbook = pd.read_csv(ORDERBOOK_FILE)
        df_orderbook["timestamp"] = pd.to_datetime(df_orderbook["timestamp"], format="%Y-%m-%d %H:%M:%S")
        logger.info(f"Loaded orderbook data: {len(df_orderbook)} rows.")
    except Exception as e:
        logger.error(f"Error loading orderbook CSV: {e}")
        df_orderbook = pd.DataFrame()
    _cached_data = (df_trades, df_extras, df_orderbook)
    _last_mod_times = mod_times
    return _cached_data

# ----------------------------
# Database Initialization
# ----------------------------

def init_db():
    """Initialize the SQLite database and create tables if they do not exist."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_label INTEGER
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_label INTEGER,
            true_label INTEGER
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized (tables predictions and feedback ensured).")

# Initialize the database at startup
init_db()

# ----------------------------
# Data Merging
# ----------------------------

def merge_data(df_trades, df_extras, df_orderbook):
    df_vib = df_extras[df_extras["symbol"] == "VIBUSDT"].copy()
    if df_vib.empty:
        logger.error("No VIBUSDT data found.")
        return None
    df_vib.sort_values("close_time", inplace=True)
    latest_candle = df_vib.iloc[-1]

    big_trades_count = len(df_trades[
        (df_trades["TradeTime"] >= latest_candle["close_time"] - timedelta(minutes=5)) &
        (df_trades["Quantity"] >= 100000)
    ])
    if not df_orderbook.empty:
        valid_snapshots = df_orderbook[df_orderbook["timestamp"] <= latest_candle["close_time"]]
        orderbook_spread = valid_snapshots.iloc[-1]["spread"] if not valid_snapshots.empty else 0.0
    else:
        orderbook_spread = 0.0

    features = [
        latest_candle.get("rsi", 0),
        latest_candle.get("macd_hist", 0),
        latest_candle.get("close", 0),
        latest_candle.get("volume", 0),
        big_trades_count,
        orderbook_spread,
        0, 0, 0  # Placeholders for diff_BTC, diff_ETH, diff_RNDR
    ]
    return np.array([features]), latest_candle["close"], latest_candle["close_time"]

# ----------------------------
# Database Write Functions
# ----------------------------

def store_prediction(timestamp, predicted_label):
    """Store a prediction into the predictions table."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO predictions (timestamp, predicted_label) VALUES (?, ?)",
                (timestamp, predicted_label))
    conn.commit()
    conn.close()

def store_feedback(timestamp, predicted_label, true_label):
    """Store feedback into the feedback table."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback (timestamp, predicted_label, true_label) VALUES (?, ?, ?)",
                (timestamp, predicted_label, true_label))
    conn.commit()
    conn.close()

# ----------------------------
# ML Feedback & Inference
# ----------------------------

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

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        resp = requests.post(url, data=data, timeout=5)
        if resp.status_code != 200:
            logger.error(f"Telegram Error: {resp.text}")
    except Exception as e:
        logger.error(f"Telegram Exception: {e}")

def process_feedback(current_vib_close, current_timestamp, model):
    global pending_predictions
    remaining_predictions = []
    for pred in pending_predictions:
        if (current_timestamp - pred["timestamp"]).total_seconds() >= FEEDBACK_WINDOW:
            try:
                pct_change = (current_vib_close - pred["vib_close"]) / pred["vib_close"]
            except Exception:
                pct_change = 0.0
            true_label = label_candle(pct_change)
            model.partial_fit(pred["features"], [true_label])
            store_feedback(pred["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), pred["predicted_label"], true_label)
            logger.info(f"[Feedback] Updated model with true label: {true_label}")
        else:
            remaining_predictions.append(pred)
    pending_predictions = remaining_predictions

def run_ml_inference(model):
    global pending_predictions
    data = load_data()
    merged_result = merge_data(*data)
    if merged_result is None:
        logger.warning("[ML] Merge returned None. Skipping inference.")
        return
    X, current_vib_close, current_timestamp = merged_result
    process_feedback(current_vib_close, current_timestamp, model)
    try:
        prediction = model.predict(X)[0]
    except Exception as e:
        logger.error(f"[ML] Error during prediction: {e}")
        return
    store_prediction(current_timestamp.strftime("%Y-%m-%d %H:%M:%S"), prediction)
    logger.info(f"[ML] Prediction: {prediction}")
    pending_predictions.append({
        "timestamp": current_timestamp,
        "features": X,
        "vib_close": current_vib_close,
        "predicted_label": prediction
    })
    if prediction != 0:
        send_telegram_alert(f"Signal: {prediction}\nFeatures: {X[0].tolist()}")

def main_loop():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Please train and save your model first.")
        return
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
    while True:
        run_ml_inference(model)
        time.sleep(30)

if __name__ == "__main__":
    main_loop()