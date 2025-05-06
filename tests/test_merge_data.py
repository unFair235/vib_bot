import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

from vib_bot.processing.utils import compute_feature_windows
from vib_bot.config import BIG_TRADE_THRESHOLD


def make_dummy_window(look_ahead: int):
    base = datetime(2021, 1, 1, 0, 0)
    # Build extras for two time points per symbol
    records = []
    for sym in ["VIBUSDT", "BTCUSDT", "ETHUSDT", "RENDERUSDT"]:
        for t in (base, base + timedelta(minutes=look_ahead)):
            records.append({
                "symbol": sym,
                "close_time": t,
                "close": 100.0,
            })
    extras = pd.DataFrame(records)
    # single big trade at t1
    trades = pd.DataFrame({
        "trade_time": [base + timedelta(minutes=look_ahead)],
        "quantity":   [BIG_TRADE_THRESHOLD],
    })
    # two orderbook snapshots
    orderbook = pd.DataFrame({
        "timestamp": [base, base + timedelta(minutes=look_ahead)],
        "spread":    [0.1, 0.2],
    })
    # vib_row = the second VIBUSDT record
    vib_row = extras[(extras.symbol == "VIBUSDT") & (extras.close_time == base + timedelta(minutes=look_ahead))].iloc[0].copy()
    # No future price change
    vib_row["future_close"] = vib_row["close"]
    return extras, trades, orderbook, vib_row


def test_compute_feature_windows_counts_and_spread():
    look_ahead = 5
    extras, trades, orderbook, vib_row = make_dummy_window(look_ahead)
    big_count, spread, dBTC, dETH, dRNDR = compute_feature_windows(
        extras, trades, orderbook, vib_row, look_ahead
    )
    assert big_count == 1, "Should count the one big trade in window"
    assert spread == 0.2, "Should pick latest orderbook spread"


def test_compute_feature_windows_ratios_default():
    look_ahead = 5
    extras, trades, orderbook, vib_row = make_dummy_window(look_ahead)
    # Since close prices don't change and future_close == close, ratios should be 1.0
    _, _, dBTC, dETH, dRNDR = compute_feature_windows(
        extras, trades, orderbook, vib_row, look_ahead
    )
    assert pytest.approx(dBTC, rel=1e-6) == 1.0
    assert pytest.approx(dETH, rel=1e-6) == 1.0
    assert pytest.approx(dRNDR, rel=1e-6) == 1.0


def test_compute_feature_windows_no_big_trades():
    # If quantity below threshold, big_count should be zero
    look_ahead = 5
    extras, trades, orderbook, vib_row = make_dummy_window(look_ahead)
    trades.quantity = [BIG_TRADE_THRESHOLD - 1]
    big_count, *_ = compute_feature_windows(extras, trades, orderbook, vib_row, look_ahead)
    assert big_count == 0, "Should be zero when no trades meet threshold"
