import os
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from vib_bot.config import MASTER_DB_FILE as ORIGINAL_MASTER_DB
from vib_bot.monitoring.check_winrate import main as check_winrate_main
from vib_bot.utils.metrics import compute_drawdowns, sharpe_ratio, trade_statistics


def test_compute_drawdowns_and_series():
    equity_curve = np.array([100.0, 120.0, 90.0, 95.0])
    result = compute_drawdowns(equity_curve)
    # peaks: [100,120,120,120]
    expected_drawdowns = (equity_curve - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve)
    assert np.allclose(result['drawdown_series'], expected_drawdowns)
    # max drawdown = (90-120)/120 = -0.25
    assert result['max_drawdown'] == pytest.approx(-0.25)


def test_sharpe_ratio():
    # returns with mean 1%, std 1% -> sharpe ~ sqrt(252)
    returns = np.array([0.01, 0.01, 0.01, 0.01])
    sr = sharpe_ratio(returns, freq=252)
    assert sr == pytest.approx(np.sqrt(252) * (0.01 / 0.01))


def test_trade_statistics():
    # Two trades: one LONG win, one SHORT win
    df = pd.DataFrame({
        'entry_time': [datetime(2021,1,1), datetime(2021,1,2)],
        'exit_time':  [datetime(2021,1,2), datetime(2021,1,3)],
        'entry_price': [100.0, 100.0],
        'exit_price':  [110.0, 90.0],
        'quantity':    [1.0, 1.0],
        'side':        ['LONG', 'SHORT'],
    })
    stats = trade_statistics(df)
    # Both trades win, so win_rate=1.0
    assert stats['win_rate'] == pytest.approx(1.0)
    # Each PnL = 10.0, so avg_return=10.0
    assert stats['average_return'] == pytest.approx(10.0)
    # losses sum = 0 -> profit_factor = inf
    assert stats['profit_factor'] == np.inf
    # average_duration = (1 day + 1 day) / 2 = 1 day
    assert stats['average_duration'] == timedelta(days=1)


def test_check_winrate_script(tmp_path, monkeypatch, capsys):
    # Create a temporary master DB with one matching record
    temp_db = tmp_path / "master.db"
    conn = sqlite3.connect(temp_db)
    # Create tables
    conn.execute("CREATE TABLE predictions(timestamp TEXT, predicted_label INTEGER);")
    conn.execute("CREATE TABLE feedback(timestamp TEXT, true_label INTEGER);")
    # Insert one matching and one non-matching
    conn.execute("INSERT INTO predictions(timestamp, predicted_label) VALUES (?, ?)",
                 ("2021-01-01T00:00:00", 2))
    conn.execute("INSERT INTO feedback(timestamp, true_label) VALUES (?, ?)",
                 ("2021-01-01T00:00:00", 2))
    # a mismatched timestamp
    conn.execute("INSERT INTO feedback(timestamp, true_label) VALUES (?, ?)",
                 ("2021-01-02T00:00:00", 1))
    conn.commit()
    conn.close()

    # Monkeypatch config to use temp_db
    monkeypatch.setattr('vib_bot.config.MASTER_DB_FILE', str(temp_db))

    # Run the script's main
    check_winrate_main()
    captured = capsys.readouterr()
    assert "Winrate (Accuracy): 100.00% over 1 samples" in captured.out
    assert "Classification Report:" in captured.out
    assert "Sample predictions vs true labels:" in captured.out

    # Restore original MASTER_DB_FILE
    monkeypatch.setattr('vib_bot.config.MASTER_DB_FILE', ORIGINAL_MASTER_DB)
