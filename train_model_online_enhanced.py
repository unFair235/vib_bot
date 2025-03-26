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
import sqlite3
import requests

# ----------------------------
# Configuration & File Paths
# ----------------------------
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

TRADES_FILE = os.path.join(BASE_DIR, "vib_trades_log.csv")
EXTRAS_FILE = os.path.join(BASE_DIR, "vib_extra_data.csv")
ORDERBOOK_FILE = os.path.join(BASE_DIR, "orderbook_data.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model_online_enhanced.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_online.pkl")

# SQLite database for storing merged training data
DB_FILE = os.path.join(BASE_DIR, "training_data.db")

# Telegram configuration (ensure these are defined)
TELEGRAM_TOKEN = "7636229600:AAESoUoIB6nIcUHxme43x8byKhX1sok5zPk"
CHAT_ID = 531265494

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants for training
BIG_TRADE_THRESHOLD = 100000
LOOKAHEAD_SECONDS = 3600  # 1 hour lookahead
FEEDBACK_WINDOW = 3600    # Feedback window (1 hour)

# ----------------------------
# Database Initialization for Training Data
# ----------------------------
def init_training_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS merged_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            rsi REAL,
            macd_hist REAL,
            vib_close REAL,
            volume REAL,
            big_trades_count INTEGER,
            orderbook_spread REAL,
            diff_BTC REAL,
            diff_ETH REAL,
            diff_RNDR REAL,
            label INTEGER
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Training database initialized (merged_training_data table ensured).")

init_training_db()

def store_training_data(df_train):
    conn = sqlite3.connect(DB_FILE)
    # Add a timestamp column indicating when data was stored
    df_train["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = ["timestamp", "rsi", "macd_hist", "vib_close", "volume", 
            "big_trades_count", "orderbook_spread", "diff_BTC", "diff_ETH", "diff_RNDR", "label"]
    df_train = df_train[cols]
    df_train.to_sql("merged_training_data", conn, if_exists="append", index=False)
    conn.close()
    logger.info(f"Merged training data stored in database with {len(df_train)} rows.")

# ----------------------------
# Data Loading & Merging Functions
# ----------------------------
def load_data():
    try:
        df_extras = pd.read_csv(EXTRAS_FILE, parse_dates=["open_time", "close_time"])
        logger.info(f"Loaded extras data: {len(df_extras)} rows.")
    except Exception as e:
        logger.error(f"Error loading extras CSV: {e}")
        df_extras = pd.DataFrame()
    try:
        df_trades = pd.read_csv(TRADES_FILE, parse_dates=["LocalTime", "TradeTime"])
        logger.info(f"Loaded trades data: {len(df_trades)} rows.")
    except Exception as e:
        logger.error(f"Error loading trades CSV: {e}")
        df_trades = pd.DataFrame()
    try:
        df_orderbook = pd.read_csv(ORDERBOOK_FILE)
        df_orderbook["timestamp"] = pd.to_datetime(df_orderbook["timestamp"], format="%Y-%m-%d %H:%M:%S")
        logger.info(f"Loaded orderbook data: {len(df_orderbook)} rows.")
    except Exception as e:
        logger.error(f"Error loading orderbook CSV: {e}")
        df_orderbook = pd.DataFrame()
    df_extras.sort_values("open_time", inplace=True)
    return df_extras, df_trades, df_orderbook

def merge_data(df_extras, df_trades, df_orderbook):
    df_vib = df_extras[df_extras["symbol"] == "VIBUSDT"].copy()
    if df_vib.empty:
        logger.error("No VIBUSDT data found.")
        return None
    df_vib.sort_values("close_time", inplace=True)
    latest_candle = df_vib.iloc[-1]
    
    big_trades_count = len(df_trades[
        (df_trades["TradeTime"] >= latest_candle["close_time"] - timedelta(minutes=5)) &
        (df_trades["Quantity"] >= BIG_TRADE_THRESHOLD)
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

def store_prediction(timestamp, predicted_label):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # Insert the prediction into the 'label' column of merged_training_data table
    cur.execute("INSERT INTO merged_training_data (timestamp, label) VALUES (?, ?)", 
                (timestamp, predicted_label))
    conn.commit()
    conn.close()

def store_feedback(timestamp, predicted_label, true_label):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_label INTEGER,
            true_label INTEGER
        );
    """)
    conn.commit()
    cur.execute("INSERT INTO feedback (timestamp, predicted_label, true_label) VALUES (?, ?, ?)",
                (timestamp, predicted_label, true_label))
    conn.commit()
    conn.close()

pending_predictions = []

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
            store_feedback(pred["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                           pred["predicted_label"], true_label)
            logger.info(f"[Feedback] Updated model with true label: {true_label}")
        else:
            remaining_predictions.append(pred)
    pending_predictions = remaining_predictions

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
    df_extras, df_trades, df_orderbook = load_data()
    merged_result = merge_data(df_extras, df_trades, df_orderbook)
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