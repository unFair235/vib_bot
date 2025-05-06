#!/usr/bin/env python3
"""
vib_bot/backtest/backtest_harness.py

Orchestrate backtests → compute features → replay predictions → P&L series → metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

import vib_bot.config as cfg
from vib_bot.utils.metrics import sharpe_ratio, profit_factor, max_drawdown
from vib_bot.backtest.backtest_utils import backtest_price_series


def compute_features_at(
    extras: pd.DataFrame,
    trades: pd.DataFrame,
    orderbook: pd.DataFrame,
    symbol: str,
    ts: pd.Timestamp
) -> Optional[np.ndarray]:
    """
    Compute the exact same 1×9 feature vector that get_features() would produce
    for a live timestamp `ts`, but using static DataFrames.
    Returns None if data is stale or insufficient.
    """
    # Data freshness check (use close_time, not open_time)
    if extras.empty:
        return None
    age = (ts - extras["close_time"].max()).total_seconds()
    if age > cfg.DATA_FRESHNESS_THRESHOLD:
        return None

    # Filter for our symbol
    sym_extras = extras[extras.symbol == symbol].sort_values("close_time")
    if sym_extras.empty or sym_extras.iloc[-1].volume == 0:
        return None

    # Find the window
    latest = sym_extras.iloc[-1]
    start = ts - pd.Timedelta(minutes=cfg.LOOK_AHEAD)
    window = sym_extras[sym_extras.close_time >= start]
    if window.empty:
        return None

    first = window.iloc[0]
    vib_pct = (latest.close - first.close) / first.close if first.close else 0.0

    # Big trades count
    big_cnt = int(
        trades[
            (trades.symbol == symbol) &
            (trades.trade_time >= start) &
            (trades.quantity >= cfg.BIG_TRADE_THRESHOLD)
        ].shape[0]
    )

    # Latest spread
    ob_sym = orderbook[orderbook.symbol == symbol]
    spread = (
        ob_sym[ob_sym.timestamp <= ts].spread.iloc[-1]
        if not ob_sym.empty else 0.0
    )

    # Cross‑symbol ratios
    ratios: Dict[str, float] = {}
    for other in cfg.SYMBOLS:
        if other == symbol:
            continue
        w = extras[
            (extras.symbol == other) &
            (extras.close_time >= start) &
            (extras.close_time <= ts)
        ]
        if len(w) >= 2:
            pct = (w.close.iloc[-1] - w.close.iloc[0]) / w.close.iloc[0]
            ratios[other] = 1.0 + (pct - vib_pct)
        else:
            ratios[other] = 1.0

    # Build feature vector
    features = np.array([[
        latest.rsi,
        latest.macd_hist,
        latest.close,
        latest.volume,
        big_cnt,
        spread,
        ratios.get("BTCUSDT", 1.0),
        ratios.get("ETHUSDT", 1.0),
        ratios.get("RENDERUSDT", 1.0),
    ]], dtype=float)

    return features


def run_backtest_for_symbol(
    extras: pd.DataFrame,
    trades: pd.DataFrame,
    orderbook: pd.DataFrame,
    symbol: str,
    model,
    scaler=None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    For a single symbol, roll through each extras.close_time, compute features,
    predict, and capture the one‑step ahead return.
    Returns:
      - rets: numpy array of strategy returns
      - metrics: dict with 'sharpe', 'profit_factor', 'max_drawdown'
    """
    sym_extras = extras[extras.symbol == symbol].sort_values("close_time")
    price_series = sym_extras.set_index("close_time")["close"]

    rets = []
    for i, ts in enumerate(price_series.index):
        feat = compute_features_at(extras, trades, orderbook, symbol, ts)
        if feat is None:
            continue

        X = feat if scaler is None else scaler.transform(feat)
        preds = model.predict(X)
        if preds.ndim == 1:
            label = float(preds[0])
        else:
            label = float(
                np.argmax(preds, axis=1)[0]
                - ((preds.shape[1] - 1) // 2)
            )

        if i + 1 >= len(price_series):
            break
        p0 = price_series.iloc[i]
        p1 = price_series.iloc[i + 1]
        rets.append(label * (p1 - p0) / p0)

    rets = np.array(rets, dtype=float)
    metrics = {
        "sharpe": sharpe_ratio(rets),
        "profit_factor": profit_factor(rets),
        "max_drawdown": max_drawdown(np.cumsum(rets)),
    }
    return rets, metrics


def run_backtests(
    extras: pd.DataFrame,
    trades: pd.DataFrame,
    orderbook: pd.DataFrame,
    models: Dict[str, Tuple]
) -> Dict[str, Dict]:
    """
    Run backtest for each model in `models` mapping id → (model, scaler).
    Returns dict of model_id → { 'returns': np.ndarray, 'metrics': {...} }
    """
    results: Dict[str, Dict] = {}
    for model_id, (model, scaler) in models.items():
        rets, metrics = run_backtest_for_symbol(
            extras, trades, orderbook, cfg.SYMBOLS[0], model, scaler
        )
        results[model_id] = {"returns": rets, "metrics": metrics}
    return results