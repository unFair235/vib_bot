#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# ----------------------------
# Configuration & File Paths
# ----------------------------
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"
TRAINING_DB_FILE = os.path.join(BASE_DIR, "training_data.db")
# We'll assume your historical training data is stored in a table named 'merged_training_data'
# with columns: timestamp, rsi, macd_hist, vib_close, volume, big_trades_count, orderbook_spread, diff_BTC, diff_ETH, diff_RNDR, label.
MODEL_PATH_NN = os.path.join(BASE_DIR, "model_nn.h5")
SCALER_PATH_NN = os.path.join(BASE_DIR, "scaler_nn.pkl")

# Define the feature and label columns
FEATURE_COLUMNS = ["rsi", "macd_hist", "vib_close", "volume", "big_trades_count", "orderbook_spread", "diff_BTC", "diff_ETH", "diff_RNDR"]
LABEL_COLUMN = "label"

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger("train_model_nn")

# ----------------------------
# Data Loading Function
# ----------------------------
def load_training_data():
    """
    Loads training data from the SQLite database.
    Expects a table 'merged_training_data' with the required feature columns and a label column.
    """
    try:
        conn = sqlite3.connect(TRAINING_DB_FILE)
        df = pd.read_sql_query("SELECT * FROM merged_training_data", conn, parse_dates=["timestamp"])
        conn.close()
        logger.info(f"Loaded {len(df)} rows of training data.")
        return df
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return pd.DataFrame()

# ----------------------------
# Model Building Function
# ----------------------------
def build_model(input_dim, num_classes):
    """
    Builds a simple feedforward neural network using Keras.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    # Assuming label classes range from -3 to +3 (7 classes)
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ----------------------------
# Main Training Function
# ----------------------------
def main():
    # Load training data from SQLite
    df = load_training_data()
    if df.empty:
        logger.error("No training data available. Exiting.")
        return

    # Drop rows with missing values in feature or label columns
    df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN], inplace=True)
    
    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_COLUMN].values

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save the scaler for later use during inference
    joblib.dump(scaler, SCALER_PATH_NN)
    logger.info("Scaler saved.")

    input_dim = X_train_scaled.shape[1]
    num_classes = len(np.unique(y))
    logger.info(f"Input dimension: {input_dim}, Number of classes: {num_classes}")

    # Build and train the model
    model = build_model(input_dim, num_classes)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    history = model.fit(X_train_scaled, y_train,
                        validation_data=(X_val_scaled, y_val),
                        epochs=100,
                        batch_size=32,
                        callbacks=[early_stop])
    
    # Evaluate model performance on validation data
    val_loss, val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    # Save the trained model to disk
    model.save(MODEL_PATH_NN)
    logger.info(f"Neural network model saved to {MODEL_PATH_NN}")

if __name__ == "__main__":
    main()