#!/usr/bin/env python3
import logging
import joblib
import numpy as np
import pandas as pd
import sqlite3
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import vib_bot.config as cfg

# ─── Feature & label columns ─────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "rsi", "macd_hist", "vib_close", "volume",
    "big_trades_count", "orderbook_spread",
    "diff_BTC", "diff_ETH", "diff_RNDR"
]
LABEL_COLUMN = "label"

# ─── Logging Setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("train_model_linear")


def load_training_data() -> pd.DataFrame:
    """Load merged_training_data from the runtime-configured SQLite."""
    try:
        conn = sqlite3.connect(cfg.TRAINING_DB_FILE)
        df = pd.read_sql_query(
            "SELECT * FROM merged_training_data",
            conn,
            parse_dates=["timestamp"]
        )
        conn.close()
        logger.info(f"Loaded {len(df)} rows of training data from {cfg.TRAINING_DB_FILE}.")
        return df
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return pd.DataFrame()


def main():
    df = load_training_data()
    if df.empty:
        logger.error("No training data available. Exiting.")
        return

    # Impute missing features
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill().fillna(0)

    # Drop rows missing the label
    df.dropna(subset=[LABEL_COLUMN], inplace=True)
    if df.empty:
        logger.error("All rows had missing labels. Exiting.")
        return

    # Extract features & labels
    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_COLUMN].astype(int).values

    # Scale & persist scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, cfg.SCALER_PATH_LINEAR)
    logger.info(f"Scaler saved to {cfg.SCALER_PATH_LINEAR}")

    # Train initial SGDClassifier via partial_fit
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        max_iter=1000,
        tol=1e-3,
        random_state=42
    )
    classes = np.unique(y)
    clf.partial_fit(X_scaled, y, classes=classes)
    logger.info("Initial partial_fit on full historical data complete.")

    # Persist the model
    joblib.dump(clf, cfg.MODEL_PATH_LINEAR)
    logger.info(f"Linear model saved to {cfg.MODEL_PATH_LINEAR}")


if __name__ == "__main__":
    main()