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

# ----------------------------
# Configuration & File Paths
# ----------------------------
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
MASTER_DB_FILE = os.path.join(BASE_DIR, "vib_master.db")
EXTRAS_DB_FILE = os.path.join(BASE_DIR, "vib_extra_data.db")

# Feedback window in seconds
FEEDBACK_WINDOW = 3600

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ----------------------------
# DB Functions for Pending Feedback
# ----------------------------
def load_pending_feedback():
    try:
        conn = sqlite3.connect(MASTER_DB_FILE)
        df = pd.read_sql_query("SELECT * FROM pending_feedback", conn, parse_dates=["timestamp"])
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error loading pending feedback: {e}")
        return pd.DataFrame()

def delete_pending_feedback(record_id):
    try:
        conn = sqlite3.connect(MASTER_DB_FILE)
        cur = conn.cursor()
        cur.execute("DELETE FROM pending_feedback WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error deleting pending feedback record {record_id}: {e}")

def store_feedback(timestamp, predicted_label, true_label):
    try:
        conn = sqlite3.connect(MASTER_DB_FILE)
        cur = conn.cursor()
        cur.execute("INSERT INTO feedback (timestamp, predicted_label, true_label) VALUES (?, ?, ?)",
                    (timestamp, predicted_label, true_label))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")

# ----------------------------
# Utility to Get Latest VIB Price from Extras DB
# ----------------------------
def get_latest_vib_price():
    try:
        conn = sqlite3.connect(EXTRAS_DB_FILE)
        df = pd.read_sql_query("SELECT * FROM vib_extra_data WHERE symbol='VIBUSDT' ORDER BY close_time DESC LIMIT 1", conn, parse_dates=["close_time"])
        conn.close()
        if not df.empty:
            return df.iloc[0]["close"]
    except Exception as e:
        logger.error(f"Error fetching latest VIB price: {e}")
    return None

# ----------------------------
# Training Update Loop
# ----------------------------
def update_model():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}.")
        return
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded for training update.")
    
    df_pending = load_pending_feedback()
    if df_pending.empty:
        logger.info("No pending feedback records to process.")
        return
    
    current_time = datetime.now()
    latest_vib_price = get_latest_vib_price()
    if latest_vib_price is None:
        logger.error("Could not retrieve latest VIB price; aborting update.")
        return
    
    updated = False
    for idx, row in df_pending.iterrows():
        record_time = row["timestamp"]
        elapsed = (current_time - record_time).total_seconds()
        if elapsed < FEEDBACK_WINDOW:
            continue  # Skip if feedback window not reached
        try:
            # Retrieve stored features from JSON
            features = np.array([json.loads(row["features"])])
            # Assume the stored 'vib_price' was the prediction price; true feedback is based on the latest VIB price.
            # Compute percent change from the stored prediction price to the current price.
            # Note: This is a proxy for actual feedback.
            # In a live system, you might compute this differently.
            # For now:
            # true_label = label based on (latest_vib_price - predicted_vib_price) / predicted_vib_price
            predicted_vib_price = features[0][2]  # Assuming the 3rd feature is the VIB close price at prediction time.
            pct_change = (latest_vib_price - predicted_vib_price) / predicted_vib_price if predicted_vib_price else 0.0
            # Determine true label using same logic as in your label_candle function:
            if pct_change >= 0.10:
                true_label = 3
            elif pct_change >= 0.05:
                true_label = 2
            elif pct_change >= 0.01:
                true_label = 1
            elif pct_change > -0.01:
                true_label = 0
            elif pct_change > -0.05:
                true_label = -1
            elif pct_change > -0.10:
                true_label = -2
            else:
                true_label = -3
            predicted_label = row["predicted_label"]
            logger.info(f"Record {row['id']}: Elapsed {elapsed:.2f}s, Predicted {predicted_label}, True {true_label}")
            # Update model with partial_fit
            model.partial_fit(features, [true_label])
            # Store feedback in DB
            store_feedback(record_time.strftime("%Y-%m-%d %H:%M:%S"), predicted_label, true_label)
            # Delete processed pending feedback
            delete_pending_feedback(row["id"])
            updated = True
        except Exception as e:
            logger.error(f"Error processing pending feedback record {row['id']}: {e}")
    
    if updated:
        try:
            joblib.dump(model, MODEL_PATH)
            logger.info("Model updated and saved to disk.")
        except Exception as e:
            logger.error(f"Error saving updated model: {e}")
    else:
        logger.info("No pending feedback records met the criteria for update.")

def main_loop():
    while True:
        update_model()
        # Run update periodically, e.g., every 10 minutes
        time.sleep(600)

if __name__ == "__main__":
    main_loop()