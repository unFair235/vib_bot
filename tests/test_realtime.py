import os
import sqlite3
import numpy as np
from datetime import datetime
import pytest

import vib_bot.config as cfg
from vib_bot.realtime.vib_master import run_cycle, get_conn

# Dummy model that always predicts label 2
class DummyModel:
    def predict(self, X):
        # return array of shape (n_samples,) or (n_samples,1)
        return np.array([2])

@pytest.fixture(autouse=True)
def setup_env(tmp_path, monkeypatch):
    # Create isolated BASE_DIR
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    monkeypatch.setenv("VIB_BOT_BASE_DIR", str(base_dir))
    # Reload config to pick up new BASE_DIR
    import importlib
    importlib.reload(cfg)
    # Ensure MASTER_DB_FILE directory
    db_path = cfg.MASTER_DB_FILE
    # Initialize empty master DB with required tables
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS predictions;")
    conn.execute("DROP TABLE IF EXISTS pending_feedback;")
    conn.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, timestamp TEXT, predicted_label INTEGER, model_id TEXT);")
    conn.execute("CREATE TABLE pending_feedback (id INTEGER PRIMARY KEY, timestamp TEXT, predicted_label INTEGER, features TEXT, model_id TEXT);")
    conn.commit()
    conn.close()
    return base_dir

def test_run_cycle_writes_to_db(setup_env, monkeypatch):
    # Stub load_active_model to use DummyModel and no scaler
    monkeypatch.setattr(
        'vib_bot.realtime.vib_master.load_active_model',
        lambda: ((DummyModel(), None), 'linear')
    )
    # Stub get_features to return a fixed feature vector, price, and timestamp
    raw_X = np.zeros((1,9))
    now = datetime.utcnow()
    monkeypatch.setattr(
        'vib_bot.realtime.vib_master.get_features',
        lambda: (raw_X, 100.0, now)
    )
    # Stub decision → LONG (ensures follow-through)
    monkeypatch.setattr(
        'vib_bot.realtime.vib_master.make_decision',
        lambda features, label: 'LONG'
    )
    # Stub sizing
    monkeypatch.setattr(
        'vib_bot.realtime.vib_master.compute_position_size',
        lambda equity, price: 1.0
    )
    # Stub risk manager to always allow
    monkeypatch.setattr(
        'vib_bot.realtime.vib_master.assess_risk',
        lambda action, price, features: (True, 99.0, 101.0)
    )
    # Stub execution and logging
    monkeypatch.setattr('vib_bot.realtime.vib_master.place_order', lambda **kw: {'order': {}, 'oco': None})
    monkeypatch.setattr('vib_bot.realtime.vib_master.log_trade', lambda **kw: None)
    monkeypatch.setattr('vib_bot.realtime.vib_master.send_telegram_alert', lambda msg: None)

    # Run one cycle
    run_cycle()

    # Verify prediction was stored
    conn = sqlite3.connect(cfg.MASTER_DB_FILE)
    preds = conn.execute("SELECT timestamp, predicted_label, model_id FROM predictions;").fetchall()
    pending = conn.execute("SELECT timestamp, predicted_label, features, model_id FROM pending_feedback;").fetchall()
    conn.close()

    assert len(preds) == 1, "Expected one prediction record"
    assert len(pending) == 1, "Expected one pending_feedback record"
    # Check values
    assert preds[0][1] == 2
    assert preds[0][2] == 'linear'
    assert pending[0][1] == 2
    assert pending[0][3] == 'linear'
