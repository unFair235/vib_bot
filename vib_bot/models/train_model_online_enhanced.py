#!/usr/bin/env python3
import os
import time
import json
import joblib
import logging
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from vib_bot.config import (
    EXTRAS_DB_FILE,
    MASTER_DB_FILE,
    FEEDBACK_WINDOW,
    MODEL_PATH_LINEAR,
    SCALER_PATH_LINEAR,
)

# ——— Paths & Constants ———
MODEL_PATH  = MODEL_PATH_LINEAR
SCALER_PATH = SCALER_PATH_LINEAR

# ——— Logging Setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("online_updater")

# ——— Schema Migration ———
def _ensure_symbol_column():
    """
    Ensure 'symbol' TEXT column exists in pending_feedback and feedback.
    """
    with sqlite3.connect(MASTER_DB_FILE) as conn:
        cur = conn.cursor()
        for table in ("pending_feedback", "feedback"):
            # Fetch existing columns
            cur.execute(f"PRAGMA table_info({table});")
            cols = [row[1] for row in cur.fetchall()]
            if "symbol" not in cols:
                logger.info(f"Adding 'symbol' column to {table}")
                cur.execute(f"ALTER TABLE {table} ADD COLUMN symbol TEXT;")
        conn.commit()

# run migration once at import
_try = False
try:
    _ensure_symbol_column()
except Exception as e:
    logger.error(f"Schema migration failed: {e}", exc_info=True)


# ——— DB Helpers ———
def load_pending_feedback() -> pd.DataFrame:
    """
    Expect pending_feedback to have columns:
      id, timestamp, predicted_label, features, model_id, symbol
    """
    try:
        with sqlite3.connect(MASTER_DB_FILE) as conn:
            df = pd.read_sql_query(
                "SELECT id, timestamp, predicted_label, features, model_id, symbol "
                "FROM pending_feedback",
                conn,
                parse_dates=["timestamp"]
            )
        return df
    except Exception as e:
        logger.error(f"Error loading pending feedback: {e}", exc_info=True)
        return pd.DataFrame()

def delete_pending_feedback(record_id: int):
    try:
        with sqlite3.connect(MASTER_DB_FILE) as conn:
            conn.execute("DELETE FROM pending_feedback WHERE id = ?", (record_id,))
            conn.commit()
    except Exception as e:
        logger.error(f"Error deleting pending feedback record {record_id}: {e}")

def store_feedback(timestamp: str, predicted_label: int, true_label: int, symbol: str):
    """
    Append into feedback(timestamp, predicted_label, true_label, symbol)
    """
    try:
        with sqlite3.connect(MASTER_DB_FILE) as conn:
            conn.execute(
                "INSERT INTO feedback(timestamp, predicted_label, true_label, symbol) "
                "VALUES (?, ?, ?, ?)",
                (timestamp, predicted_label, true_label, symbol)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error storing feedback: {e}", exc_info=True)


# ——— Price Lookup ———
def get_latest_price(symbol: str) -> Optional[float]:
    """
    Grab the most recent close for `symbol` from vib_extra_data.
    """
    try:
        with sqlite3.connect(EXTRAS_DB_FILE) as conn:
            df = pd.read_sql_query(
                "SELECT close FROM vib_extra_data "
                "WHERE symbol = ? "
                "ORDER BY close_time DESC LIMIT 1",
                conn,
                params=(symbol,),
                parse_dates=["close_time"]
            )
        if not df.empty:
            return float(df.iloc[0]["close"])
    except Exception as e:
        logger.error(f"Error fetching latest price for {symbol}: {e}", exc_info=True)
    return None


# ——— Post‐update Evaluation ———
def evaluate_model_performance():
    try:
        with sqlite3.connect(MASTER_DB_FILE) as conn:
            df_fb = pd.read_sql_query(
                "SELECT true_label, predicted_label FROM feedback",
                conn
            )
        if not df_fb.empty:
            acc = accuracy_score(df_fb["true_label"], df_fb["predicted_label"])
            logger.info("Post-update model accuracy: %.4f", acc)
        else:
            logger.info("No feedback rows to evaluate.")
    except Exception as e:
        logger.error(f"Error evaluating model performance: {e}", exc_info=True)


# ——— Core Update Loop ———
def update_model():
    # 1) Ensure we have a model & scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        logger.error("Model or scaler missing; skipping update.")
        return

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Loaded linear model + scaler for incremental update.")

    # 2) Pull all pending feedback for the linear model
    df_pending = load_pending_feedback()
    df_lin     = df_pending[df_pending["model_id"] == "linear"]
    if df_lin.empty:
        logger.info("No pending feedback for linear model.")
        return

    now = datetime.utcnow()
    updated = False

    for _, row in df_lin.iterrows():
        rec_time = row["timestamp"].to_pydatetime()
        # only process once FEEDBACK_WINDOW has elapsed
        if (now - rec_time).total_seconds() < FEEDBACK_WINDOW:
            continue

        symbol = row["symbol"]
        latest = get_latest_price(symbol)
        if latest is None:
            logger.error(f"Could not retrieve latest price for {symbol}; skipping record {row['id']}.")
            continue

        try:
            feats = np.array(json.loads(row["features"]))
            # flatten special cases
            if feats.ndim == 3 and feats.shape[1] == 1:
                feats = feats.reshape(1, feats.shape[2])
            if feats.shape != (1, 9):
                logger.error(f"Record {row['id']} wrong feature shape {feats.shape}; dropping.")
                delete_pending_feedback(row["id"])
                continue

            Xs = scaler.transform(feats)
            entry_price = float(feats[0][2])
            pct_change = (latest - entry_price) / entry_price if entry_price else 0.0

            # bucket to [-3..+3]
            if   pct_change >= 0.10: true_label =  3
            elif pct_change >= 0.05: true_label =  2
            elif pct_change >= 0.01: true_label =  1
            elif pct_change >  -0.01: true_label = 0
            elif pct_change >  -0.05: true_label = -1
            elif pct_change >  -0.10: true_label = -2
            else:                   true_label = -3

            logger.info(
                "Updating id %d [%s]: pred=%d → true=%d (Δ%.2f%%)",
                row["id"], symbol, row["predicted_label"], true_label, pct_change*100
            )

            # partial_fit and store
            model.partial_fit(Xs, [true_label])
            store_feedback(
                row["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                int(row["predicted_label"]),
                true_label,
                symbol
            )
            delete_pending_feedback(row["id"])
            updated = True

        except Exception as e:
            logger.error(f"Error processing record {row['id']}: {e}", exc_info=True)

    # 3) If we updated at least one, persist and re‐evaluate
    if updated:
        joblib.dump(model, MODEL_PATH)
        logger.info("Model incrementally updated & saved.")
        evaluate_model_performance()
    else:
        logger.info("No pending records ready for update.")


def main_loop():
    while True:
        update_model()
        time.sleep(600)  # every 10 minutes


if __name__ == "__main__":
    main_loop()