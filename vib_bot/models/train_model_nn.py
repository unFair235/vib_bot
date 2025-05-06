#!/usr/bin/env python3
import sqlite3
import pandas as pd
import joblib
import logging

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import vib_bot.config as cfg

# Feature columns (must match your merged_training_data)
FEATURE_COLUMNS = [
    "rsi", "macd_hist", "vib_close", "volume",
    "big_trades_count", "orderbook_spread",
    "diff_BTC", "diff_ETH", "diff_RNDR"
]
# Raw label in DB is in [-3..+3]
LABEL_COLUMN     = "label"
# Shift to [0..6]
LABEL_INDEX_NAME = "label_idx"

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("train_model_nn")


def load_training_data() -> pd.DataFrame:
    """Loads merged_training_data (including raw label) from SQLite."""
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


def build_model(input_dim: int, num_classes: int) -> Sequential:
    """Simple feed‑forward NN with softmax output."""
    model = Sequential([
        Dense(64,  input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    df = load_training_data()
    if df.empty or len(df) < 2:
        logger.warning("Not enough data (need ≥2 rows). Exiting.")
        return

    # 1) Drop any rows missing features or the raw label
    df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN], inplace=True)
    if df.empty or len(df) < 2:
        logger.warning("After dropna, not enough data. Exiting.")
        return

    # 2) Shift raw label [-3..+3] → [0..6]
    df[LABEL_INDEX_NAME] = df[LABEL_COLUMN].astype(int) + 3

    # 3) Drop the original label column so we never confuse Keras
    df.drop(columns=[LABEL_COLUMN], inplace=True)

    # 4) Extract features & new label
    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_INDEX_NAME].values

    # 5) Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6) Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # Persist the scaler
    joblib.dump(scaler, cfg.SCALER_PATH_NN)
    logger.info(f"Scaler saved to {cfg.SCALER_PATH_NN}")

    # 7) Build, train & save the model
    input_dim   = X_train_scaled.shape[1]
    num_classes = 7  # fixed for buckets -3..+3
    logger.info(f"Input dim: {input_dim}, classes: {num_classes}")

    model = build_model(input_dim, num_classes)
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )

    loss, acc = model.evaluate(X_val_scaled, y_val, verbose=0)
    logger.info(f"Validation accuracy: {acc:.4f}")

    model.save(cfg.MODEL_PATH_NN)
    logger.info(f"Neural network model saved to {cfg.MODEL_PATH_NN}")


if __name__ == "__main__":
    main()
