import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import pytest

import vib_bot.config as cfg
from tensorflow.keras.models import load_model

# Modules under test
import vib_bot.models.train_model_linear as linear_mod
import vib_bot.models.train_model_nn     as nn_mod

# Use a fixed RNG for reproducibility
rng = np.random.default_rng(42)

def create_dummy_training_db(path):
    """
    Create a training DB with a small merged_training_data for testing.
    """
    conn = sqlite3.connect(path)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2021-01-01', periods=10, freq='T'),
        'rsi': rng.random(10) * 100,
        'macd_hist': rng.standard_normal(10),
        'vib_close': rng.random(10) * 100,
        'volume': rng.random(10) * 10,
        'big_trades_count': rng.integers(0, 5, 10),
        'orderbook_spread': rng.random(10),
        'diff_BTC': rng.random(10),
        'diff_ETH': rng.random(10),
        'diff_RNDR': rng.random(10),
        'label': rng.integers(-3, 4, 10)
    })
    df.to_sql('merged_training_data', conn, index=False, if_exists='replace')
    conn.close()

@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    """
    Create isolated BASE_DIR and adjust config paths to use temp directories.
    """
    base = tmp_path / 'vib_bot_base'
    base.mkdir()
    monkeypatch.setenv('VIB_BOT_BASE_DIR', str(base))
    import importlib; importlib.reload(cfg)
    # Create necessary subdirectories
    (base / 'models').mkdir()
    return base


def test_linear_model_training_and_prediction(isolate_env):
    base = isolate_env
    # Create dummy training DB
    training_db = base / 'training_data.db'
    create_dummy_training_db(str(training_db))
    # Point config to the dummy DB
    cfg.TRAINING_DB_FILE = str(training_db)

    # Run training
    linear_mod.main()

    model_path = os.path.join(str(base), 'models', 'model.pkl')
    scaler_path = os.path.join(str(base), 'models', 'scaler_linear.pkl')
    assert os.path.exists(model_path)
    assert os.path.exists(scaler_path)

    # Load and test prediction
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(rng.random((3, 9)))
    preds = clf.predict(X_test)
    assert preds.shape == (3,)


def test_nn_model_training_and_prediction(isolate_env):
    base = isolate_env
    # Create dummy training DB
    training_db = base / 'training_data.db'
    create_dummy_training_db(str(training_db))
    # Point config to the dummy DB
    cfg.TRAINING_DB_FILE = str(training_db)

    # Run training
    nn_mod.main()

    model_path = os.path.join(str(base), 'models', 'model_nn.keras')
    scaler_path = os.path.join(str(base), 'models', 'scaler_nn.pkl')
    assert os.path.exists(model_path)
    assert os.path.exists(scaler_path)

    # Load and test prediction
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(rng.random((4, 9)))
    probs = model.predict(X_test)
    # Expect 7 output classes
    assert probs.shape == (4, 7)
