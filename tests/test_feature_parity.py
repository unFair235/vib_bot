import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

import vib_bot.config as cfg
from vib_bot.realtime.vib_master import get_features
from vib_bot.backtest.backtest_harness import compute_features_at


@pytest.fixture
def sample_dfs():
    """
    Build tiny in‐memory DataFrames for one symbol, one timestamp.
    """
    now = datetime.now(timezone.utc)
    sym = cfg.SYMBOLS[0]

    # extras: two points bracketing now
    extras = pd.DataFrame([
        {
            "symbol": sym,
            "open_time": now - timedelta(minutes=10),
            "close_time": now - timedelta(minutes=5),
            "rsi": 30,
            "macd_hist": 0.5,
            "close":  100.0,
            "volume": 200.0
        },
        {
            "symbol": sym,
            "open_time": now - timedelta(minutes=5),
            "close_time": now,
            "rsi": 35,
            "macd_hist": 0.7,
            "close":  105.0,
            "volume": 250.0
        },
    ])
    extras["open_time"]  = pd.to_datetime(extras.open_time, utc=True)
    extras["close_time"] = pd.to_datetime(extras.close_time, utc=True)

    # trades: one big trade within window
    trades = pd.DataFrame([{
        "symbol": sym,
        "trade_time": now - timedelta(minutes=3),
        "quantity": cfg.BIG_TRADE_THRESHOLD + 1
    }])
    trades["trade_time"] = pd.to_datetime(trades.trade_time, utc=True)

    # orderbook: one snapshot near now
    orderbook = pd.DataFrame([{
        "symbol": sym,
        "timestamp": now - timedelta(minutes=1),
        "spread": 0.02
    }])
    orderbook["timestamp"] = pd.to_datetime(orderbook.timestamp, utc=True)

    return extras, trades, orderbook, now, sym


def test_feature_parity(monkeypatch, sample_dfs):
    extras_df, trades_df, orderbook_df, now, sym = sample_dfs

    # Monkey‐patch the three loaders in vib_master to return our sample DFs
    monkeypatch.setattr("vib_bot.realtime.vib_master.load_extras", lambda: extras_df)
    monkeypatch.setattr("vib_bot.realtime.vib_master.load_trades", lambda: trades_df)
    monkeypatch.setattr("vib_bot.realtime.vib_master.load_orderbook", lambda: orderbook_df)

    # Call the live code
    live = get_features(sym)
    assert live is not None, "get_features returned None but sample data should be valid"
    raw_X, price, ts = live

    # Call the backtest harness
    back = compute_features_at(extras_df, trades_df, orderbook_df, sym, ts)
    assert back is not None

    # They must match exactly
    np.testing.assert_allclose(raw_X, back, rtol=1e-6, atol=1e-8)