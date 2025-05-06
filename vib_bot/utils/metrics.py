# utils/metrics.py

import numpy as np
import pandas as pd

def compute_drawdowns(equity_curve: np.ndarray) -> dict:
    """
    Computes drawdown metrics from an equity curve:
    - max_drawdown: largest drop from peak to trough
    - drawdown_series: time series of drawdowns
    """
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    max_dd = drawdowns.min()
    return {
        "drawdown_series": drawdowns,
        "max_drawdown": max_dd
    }

def sharpe_ratio(returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculates the annualized Sharpe ratio given a returns series.
    Assumes returns are period returns (e.g., daily), freq periods per year.

    Fallbacks to root-mean-square volatility if standard deviation is zero.
    """
    # mean return
    mean_r = np.nanmean(returns)
    # population standard deviation
    std_r = np.nanstd(returns)
    # fallback to RMS volatility if no variation
    if std_r == 0:
        std_r = np.sqrt(np.nanmean(np.square(returns)))
    if std_r == 0:
        return 0.0
    return np.sqrt(freq) * (mean_r / std_r)


def trade_statistics(trades: pd.DataFrame) -> dict:
    """
    Given a DataFrame of trades with columns:
      - entry_time, exit_time, entry_price, exit_price, quantity, side
    Returns:
      - win_rate
      - average_return
      - profit_factor (gross wins/gross losses)
      - average_duration (timedelta)
    """
    if trades.empty:
        return {}
    # P/L per trade
    pnl = (trades.exit_price - trades.entry_price) * trades.quantity * np.where(
        trades.side == "LONG", 1, -1
    )
    wins = pnl[pnl > 0]
    losses = -pnl[pnl < 0]

    win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0.0
    avg_return = pnl.mean() if len(pnl) > 0 else 0.0
    profit_factor = wins.sum() / losses.sum() if losses.sum() > 0 else np.inf

    durations = trades.exit_time - trades.entry_time
    avg_duration = durations.mean() if not durations.empty else pd.Timedelta(0)

    return {
        "win_rate": win_rate,
        "average_return": avg_return,
        "profit_factor": profit_factor,
        "average_duration": avg_duration
    }

def profit_factor(returns: np.ndarray) -> float:
    """
    Gross wins / gross losses from a returns series.
    """
    wins   = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return wins / losses if losses > 0 else float("inf")


def max_drawdown(cum_returns: np.ndarray) -> float:
    """
    Peak‐to‐trough drawdown from a cumulative‐returns series.
    """
    return compute_drawdowns(cum_returns)["max_drawdown"]
