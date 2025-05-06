#!/usr/bin/env python3
import os
import logging
import sqlite3
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from vib_bot.config import (
    TRAINING_DB_FILE,
    ACTIVE_MODEL_FILE,
    MODEL_PATH_LINEAR,
    SCALER_PATH_LINEAR,
    MODEL_PATH_NN,
    SCALER_PATH_NN,
)

# ——— Paths & Config ———
TRAINING_DB   = TRAINING_DB_FILE      # now pulled from config
MODEL_LINEAR  = MODEL_PATH_LINEAR
SCALER_LINEAR = SCALER_PATH_LINEAR
MODEL_NN      = MODEL_PATH_NN
SCALER_NN     = SCALER_PATH_NN
ACTIVE_FILE   = ACTIVE_MODEL_FILE

# ——— Logging Setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("eval_switch")


def load_data():
    """
    Load merged training features and labels from SQLite.
    """
    conn = sqlite3.connect(TRAINING_DB)
    df = pd.read_sql_query(
        "SELECT * FROM merged_training_data",
        conn,
        parse_dates=["timestamp"]
    )
    conn.close()
    df = df.dropna()
    feature_cols = [
        "rsi", "macd_hist", "vib_close", "volume",
        "big_trades_count", "orderbook_spread",
        "diff_BTC", "diff_ETH", "diff_RNDR"
    ]
    X = df[feature_cols].values
    y = df["label"].astype(int).values
    return X, y


def score_model(model, scaler, X, y):
    """
    Compute accuracy, applying scaler if provided and handling NN output mapping.
    For NN: argmax → [0..6], then subtract 3 → original [–3..+3].
    """
    if scaler is not None:
        X = scaler.transform(X)
    preds = model.predict(X)
    if preds.ndim > 1:
        preds = np.argmax(preds, axis=1) - 3
    return accuracy_score(y, preds)


def main():
    X, y = load_data()

    # Score linear model
    try:
        lin_model = joblib.load(MODEL_LINEAR)
        lin_scaler = joblib.load(SCALER_LINEAR)
        lin_acc = score_model(lin_model, lin_scaler, X, y)
        logger.info(f"Linear accuracy: {lin_acc:.4f}")
    except Exception as e:
        logger.error(f"Failed to evaluate linear model: {e}")
        lin_acc = None

    # Score neural network
    try:
        nn_model = load_model(MODEL_NN)
        nn_scaler = joblib.load(SCALER_NN)
        nn_acc = score_model(nn_model, nn_scaler, X, y)
        logger.info(f"NN accuracy:     {nn_acc:.4f}")
    except Exception as e:
        logger.error(f"Failed to evaluate NN model: {e}")
        nn_acc = None

    # Determine best
    best = None
    if lin_acc is not None and nn_acc is not None:
        best = "nn" if nn_acc > lin_acc else "linear"
    elif lin_acc is not None:
        best = "linear"
    elif nn_acc is not None:
        best = "nn"

    if best:
        try:
            current = None
            if os.path.exists(ACTIVE_FILE):
                with open(ACTIVE_FILE) as f:
                    current = f.read().strip()

            if current != best:
                with open(ACTIVE_FILE, "w") as f:
                    f.write(best)
                logger.info(f"Switched active model: {current} → {best}")
            else:
                logger.info(f"Active model unchanged: {current}")
        except Exception as e:
            logger.error(f"Error writing active model file: {e}")
    else:
        logger.warning("No valid accuracy scores to decide active model.")


if __name__ == "__main__":
    main()