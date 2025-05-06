import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

from vib_bot.processing.generate_training_data import build_dataset


def make_dummy_data(look_ahead: int):
    # Create two candles exactly look_ahead minutes apart
    base = datetime(2021, 1, 1, 0, 0)
    extras = pd.DataFrame({
        'symbol':       ['VIBUSDT', 'VIBUSDT'],
        'open_time':    [base, base + timedelta(minutes=look_ahead)],
        'close_time':   [base, base + timedelta(minutes=look_ahead)],
        'close':        [100.0, 110.0],
        'volume':       [10.0, 20.0],
        'rsi':          [50.0, 55.0],
        'macd_hist':    [0.1, 0.2],
    })

    trades = pd.DataFrame({
        'trade_time': [base + timedelta(minutes=look_ahead)],
        'quantity':   [1000.0]
    })

    orderbook = pd.DataFrame({
        'timestamp': [base + timedelta(minutes=look_ahead)],
        'spread':    [0.5]
    })

    return extras, trades, orderbook


def test_build_dataset_shape_and_columns():
    look_ahead = 5
    extras, trades, orderbook = make_dummy_data(look_ahead)
    df = build_dataset(extras, trades, orderbook, look_ahead)

    # Expect a single row
    assert df.shape[0] == 1, "Should generate exactly one training row"

    # Check expected columns
    expected_cols = [
        'timestamp', 'rsi', 'macd_hist', 'vib_close', 'volume',
        'big_trades_count', 'orderbook_spread',
        'diff_BTC', 'diff_ETH', 'diff_RNDR', 'label'
    ]
    assert list(df.columns) == expected_cols, f"Columns mismatch: {df.columns.tolist()}"


def test_label_and_timestamp_correctness():
    look_ahead = 5
    extras, trades, orderbook = make_dummy_data(look_ahead)
    df = build_dataset(extras, trades, orderbook, look_ahead)
    row = df.iloc[0]

    # Timestamp should match original close_time of the first candle
    assert row['timestamp'] == extras.loc[0, 'close_time'], "Timestamp should be the candle close_time"

    # Close went from 100 to 110 => 10% => bucket_label should yield 3
    assert row['label'] == 3, f"Expected label 3 for 10% increase, got {row['label']}"


def test_no_lookahead_bias_on_extra_rows():
    # If extras has extra future rows beyond look_ahead, they should not bias earlier rows
    look_ahead = 5
    extras, trades, orderbook = make_dummy_data(look_ahead)
    # Add an extra future candle far beyond look_ahead
    far_future = extras.copy()
    far_future.loc[0, 'close_time'] = extras.loc[0, 'close_time'] + timedelta(minutes=look_ahead * 10)
    far_future.loc[0, 'close'] = 200.0
    extras_extended = pd.concat([extras, far_future], ignore_index=True)

    df = build_dataset(extras_extended, trades, orderbook, look_ahead)
    # Still only one row and label based on immediate look_ahead
    assert df.shape[0] == 1
    assert df.iloc[0]['label'] == 3
