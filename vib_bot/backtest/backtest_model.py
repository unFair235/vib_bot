#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import logging

from vib_bot.config import (
    BASE_DIR,
    MODEL_PATH_LINEAR,
    SCALER_PATH_LINEAR,
    MODEL_PATH_NN,
    SCALER_PATH_NN,
)

# Path for your training data DB
TRAINING_DB_FILE = os.path.join(BASE_DIR, "training_data.db")

# ——— Logging Setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_data():
    """Load the merged_training_data table from SQLite."""
    try:
        conn = sqlite3.connect(TRAINING_DB_FILE)
        df = pd.read_sql_query(
            "SELECT * FROM merged_training_data ORDER BY timestamp",
            conn, parse_dates=["timestamp"]
        )
        conn.close()
        logger.info(f"Loaded {len(df)} rows from training DB")
        return df
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return pd.DataFrame()


def load_models():
    """Load both Linear and NN models (plus their scalers)."""
    lin, lin_scaler = None, None
    try:
        lin = joblib.load(MODEL_PATH_LINEAR)
        lin_scaler = joblib.load(SCALER_PATH_LINEAR)
        logger.info("Loaded linear model + scaler")
    except Exception as e:
        logger.error(f"Error loading linear model/scaler: {e}")

    nn, nn_scaler = None, None
    try:
        nn = load_model(MODEL_PATH_NN)
        nn_scaler = joblib.load(SCALER_PATH_NN)
        logger.info("Loaded neural‑net model + scaler")
    except Exception as e:
        logger.error(f"Error loading neural‑net model/scaler: {e}")

    return (lin, lin_scaler), (nn, nn_scaler)


def backtest(model, scaler, X):
    """
    Run model.predict on X (with optional scaler) and return integer labels.
      - Linear model: .predict(X) returns shape (n,) of -3..+3.
      - NN model:   .predict(X) returns (n,7) softmax; take argmax then subtract 3.
    """
    if model is None:
        return None
    X_in = scaler.transform(X) if scaler is not None else X
    preds = model.predict(X_in)
    if preds.ndim > 1:
        # Softmax probabilities → class index 0..6 → shift back to –3..+3
        labels = np.argmax(preds, axis=1) - 3
    else:
        labels = preds.astype(int)
    return labels


def main():
    df = load_data()
    if df.empty:
        logger.error("No data available for backtest – exiting.")
        return

    feature_cols = [
        "rsi", "macd_hist", "vib_close", "volume",
        "big_trades_count", "orderbook_spread",
        "diff_BTC", "diff_ETH", "diff_RNDR"
    ]
    X = df[feature_cols].values
    y_true = df["label"].astype(int).values

    (lin, lin_scaler), (nn, nn_scaler) = load_models()

    # Linear model backtest
    if lin is not None:
        y_lin = backtest(lin, lin_scaler, X)
        if y_lin is not None:
            acc_lin = accuracy_score(y_true, y_lin)
            logger.info("=== Linear Model ===")
            logger.info(f"Accuracy: {acc_lin:.4f}")
            print(classification_report(y_true, y_lin))

    # Neural‑net backtest
    if nn is not None:
        y_nn = backtest(nn, nn_scaler, X)
        if y_nn is not None:
            acc_nn = accuracy_score(y_true, y_nn)
            logger.info("=== Neural‑Net Model ===")
            logger.info(f"Accuracy: {acc_nn:.4f}")
            print(classification_report(y_true, y_nn))


if __name__ == "__main__":
    main()