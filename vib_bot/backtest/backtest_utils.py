# vib_bot/backtest/backtest_utils.py

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import timedelta

import vib_bot.config as cfg


def apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    side = side.upper()
    if side == "LONG":
        return price * (1.0 + slippage_pct)
    elif side == "SHORT":
        return price * (1.0 - slippage_pct)
    return price


def apply_fees(notional: float, fee_pct: float) -> float:
    return abs(notional) * fee_pct


def compute_trade_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str,
    fee_pct: float,
    slippage_pct: float
) -> float:
    entry_px = apply_slippage(entry_price, side, slippage_pct)
    exit_side = "SHORT" if side.upper() == "LONG" else "LONG"
    exit_px = apply_slippage(exit_price, exit_side, slippage_pct)

    if side.upper() == "LONG":
        gross_pnl = (exit_px - entry_px) * quantity
    else:
        gross_pnl = (entry_px - exit_px) * quantity

    fee_entry = apply_fees(entry_px * quantity, fee_pct)
    fee_exit  = apply_fees(exit_px * quantity, fee_pct)
    return gross_pnl - (fee_entry + fee_exit)


def simulate_trades(
    predictions: pd.DataFrame,
    price_series: pd.DataFrame,
    lookahead: timedelta,
    fee_pct: float,
    slippage_pct: float,
    quantity: float = 1.0
) -> pd.DataFrame:
    trades: List[Dict] = []
    ps = price_series.sort_index()

    for _, row in predictions.iterrows():
        ts    = row["timestamp"]
        label = row["predicted_label"]

        if label >= 2:
            side = "LONG"
        elif label <= -2:
            side = "SHORT"
        else:
            continue

        entry_time = ts
        exit_time  = ts + lookahead
        entry_price = ps["close"].asof(entry_time)
        exit_price  = ps["close"].asof(exit_time)
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        pnl = compute_trade_pnl(
            entry_price, exit_price, quantity,
            side, fee_pct, slippage_pct
        )
        trades.append({
            "entry_time": entry_time,
            "exit_time":  exit_time,
            "side":       side,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "quantity":    quantity,
            "pnl":         pnl
        })

    return pd.DataFrame(trades)


def compute_equity_curve(
    trades: pd.DataFrame,
    initial_equity: float
) -> pd.Series:
    trades = trades.sort_values("exit_time")
    equity = initial_equity
    curve = []
    times = []
    for _, t in trades.iterrows():
        equity += t["pnl"]
        curve.append(equity)
        times.append(t["exit_time"])
    return pd.Series(curve, index=pd.to_datetime(times), name="equity")


# ────────────────────────────────────────────────────────────────────────────────
# New all‑in‑one helper that your harness expects
# ────────────────────────────────────────────────────────────────────────────────
def backtest_price_series(
    predictions: pd.DataFrame,
    price_series: pd.Series,
    lookahead: timedelta = timedelta(minutes=cfg.LOOK_AHEAD),
    fee_pct: float   = cfg.BACKTEST_FEES_PCT,
    slippage_pct: float = cfg.BACKTEST_SLIPPAGE_PCT,
    quantity: float = 1.0,
    initial_equity: float = 1.0
) -> pd.Series:
    """
    Given a DataFrame `predictions` with ['timestamp','predicted_label']
    and a close‐price Series, simulate trades and return an equity curve.
    """
    trades_df = simulate_trades(
        predictions,
        price_series.to_frame(name="close"),
        lookahead,
        fee_pct,
        slippage_pct,
        quantity
    )
    return compute_equity_curve(trades_df, initial_equity)