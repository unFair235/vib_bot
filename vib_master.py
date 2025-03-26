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
import json

# ----------------------------
# Configuration & File Paths
# ----------------------------
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

# Database files for trades, extras, and orderbook data
TRADES_DB_FILE = os.path.join(BASE_DIR, "trades.db")
EXTRAS_DB_FILE = os.path.join(BASE_DIR, "vib_extra_data.db")
ORDERBOOK_DB_FILE = os.path.join(BASE_DIR, "orderbook.db")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Master DB file for storing predictions, feedback, and pending feedback
MASTER_DB_FILE = os.path.join(BASE_DIR, "vib_master.db")

# Telegram Bot Configuration
TELEGRAM_TOKEN = "7636229600:AAESoUoIB6nIcUHxme43x8byKhX1sok5zPk"
CHAT_ID = 531265494

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
# Global Variables
# ----------------------------
pending_predictions = []  # Global list for storing pending predictions

# ----------------------------
# Database Initialization (Master DB)
# ----------------------------
def init_db():
    conn = sqlite3.connect(MASTER_DB_FILE)
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pending_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_label INTEGER,
            features TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Master database initialized (tables predictions, feedback, and pending_feedback ensured).")

init_db()

# ----------------------------
# Database Write Functions (Master DB)
# ----------------------------
def store_prediction(timestamp, predicted_label):
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO predictions (timestamp, predicted_label) VALUES (?, ?)",
                (timestamp, predicted_label))
    conn.commit()
    conn.close()

def store_feedback(timestamp, predicted_label, true_label):
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback (timestamp, predicted_label, true_label) VALUES (?, ?, ?)",
                (timestamp, predicted_label, true_label))
    conn.commit()
    conn.close()

def store_pending_feedback(timestamp, predicted_label, features):
    features_json = json.dumps(features.tolist())
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO pending_feedback (timestamp, predicted_label, features) VALUES (?, ?, ?)",
                (timestamp, predicted_label, features_json))
    conn.commit()
    conn.close()

def delete_pending_feedback(record_id):
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    cur.execute("DELETE FROM pending_feedback WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

# ----------------------------
# Data Loading Functions (from SQLite Databases)
# ----------------------------
def load_trades_data():
    try:
        conn = sqlite3.connect(TRADES_DB_FILE)
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()
        # Convert the trade_time and local_time columns to datetime
        df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
        df["local_time"] = pd.to_datetime(df["local_time"], errors="coerce")
        logger.info(f"Loaded trades data from DB: {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error loading trades data from DB: {e}")
        df = pd.DataFrame()
    return df

def load_extras_data():
    try:
        conn = sqlite3.connect(EXTRAS_DB_FILE)
        df = pd.read_sql_query("SELECT * FROM vib_extra_data", conn, parse_dates=["open_time", "close_time"])
        conn.close()
        logger.info(f"Loaded extras data from DB: {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error loading extras data from DB: {e}")
        df = pd.DataFrame()
    return df

def load_orderbook_data():
    try:
        conn = sqlite3.connect(ORDERBOOK_DB_FILE)
        df = pd.read_sql_query("SELECT * FROM orderbook_data", conn, parse_dates=["timestamp"])
        conn.close()
        df.sort_values("timestamp", inplace=True)
        logger.info(f"Loaded orderbook data from DB: {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error loading orderbook data from DB: {e}")
        df = pd.DataFrame()
    return df

def load_data():
    df_trades = load_trades_data()
    df_extras = load_extras_data()
    df_orderbook = load_orderbook_data()
    if not df_extras.empty:
        df_extras.sort_values("open_time", inplace=True)
    return df_trades, df_extras, df_orderbook

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
    logger.info(f"Latest VIB candle: {latest_candle.to_dict()}")

    # Determine the trade time column name dynamically:
    trade_time_col = None
    for col in df_trades.columns:
        if col.lower().replace("_", "") == "tradetime":
            trade_time_col = col
            break
    if trade_time_col is None:
        logger.error("No trade time column found in trades data.")
        return None

    big_trades_count = len(df_trades[
        (df_trades[trade_time_col] >= latest_candle["close_time"] - timedelta(minutes=5)) &
        (df_trades["quantity"] >= 100000)
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
# ML Feedback & Inference Functions
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

def process_feedback(current_vib_close, current_timestamp, model):
    global pending_predictions
    remaining_predictions = []
    for pred in pending_predictions:
        elapsed = (current_timestamp - pred["timestamp"]).total_seconds()
        if elapsed >= FEEDBACK_WINDOW:
            try:
                pct_change = (current_vib_close - pred["vib_close"]) / pred["vib_close"]
            except Exception:
                pct_change = 0.0
            true_label = label_candle(pct_change)
            logger.info(f"[Feedback] Elapsed: {elapsed:.2f}s, % Change: {pct_change:.4f}, Predicted: {pred['predicted_label']}, True: {true_label}")
            model.partial_fit(pred["features"], [true_label])
            store_feedback(pred["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                           pred["predicted_label"], true_label)
        else:
            remaining_predictions.append(pred)
    pending_predictions = remaining_predictions

def update_model_from_db_feedback(model):
    conn = sqlite3.connect(MASTER_DB_FILE)
    df = pd.read_sql_query("SELECT * FROM pending_feedback", conn, parse_dates=["timestamp"])
    conn.close()
    if df.empty:
        logger.info("No pending feedback records found in DB.")
        return
    now = datetime.now()
    for idx, row in df.iterrows():
        record_time = row["timestamp"]
        if not isinstance(record_time, datetime):
            record_time = pd.to_datetime(record_time)
        elapsed = (now - record_time).total_seconds()
        if elapsed >= FEEDBACK_WINDOW:
            try:
                features = np.array([json.loads(row["features"])])
                logger.info(f"[DB Feedback] Record {row['id']}: Elapsed: {elapsed:.2f}s, Predicted: {row['predicted_label']}")
            except Exception as e:
                logger.error(f"Error processing DB feedback record {row['id']}: {e}")
            delete_pending_feedback(row["id"])
    logger.info("Processed and cleared DB feedback records.")

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        resp = requests.post(url, data=data, timeout=5)
        if resp.status_code != 200:
            logger.error(f"Telegram Error: {resp.text}")
    except Exception as e:
        logger.error(f"Telegram Exception: {e}")

def run_ml_inference(model):
    global pending_predictions
    df_trades, df_extras, df_orderbook = load_data()
    merged_result = merge_data(df_trades, df_extras, df_orderbook)
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
    store_pending_feedback(current_timestamp.strftime("%Y-%m-%d %H:%M:%S"), prediction, X)
    if prediction != 0:
        send_telegram_alert(f"Signal: {prediction}\nFeatures: {X[0].tolist()}")

def main_loop():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Please train and save your model first.")
        return
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
    update_model_from_db_feedback(model)
    while True:
        run_ml_inference(model)
        time.sleep(30)

if __name__ == "__main__":
    main_loop()