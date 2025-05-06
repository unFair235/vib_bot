#!/usr/bin/env python3
"""
monitoring/metrics_collector.py

Push live equity, Sharpe ratio, and drawdown metrics to Prometheus and also persist
into a local SQLite metrics DB for historical analysis.
"""
import time
import sqlite3
import numpy as np
import logging
from prometheus_client import start_http_server, Gauge
from vib_bot.config import MASTER_DB_FILE, METRICS_DB_FILE
from vib_bot.realtime.trade_tracker import get_equity_curve
from vib_bot.utils.metrics import compute_drawdowns, sharpe_ratio

# ─── Logging Setup ─────────────────────────────────
logger = logging.getLogger("metrics_collector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# ─── Prometheus Gauges ──────────────────────────────
EQUITY_GAUGE   = Gauge('trading_current_equity', 'Current account equity in USDT')
DRAWDOWN_GAUGE = Gauge('trading_max_drawdown', 'Maximum portfolio drawdown (negative means drawdown)')
SHARPE_GAUGE   = Gauge('trading_sharpe_ratio', 'Annualized Sharpe ratio of equity returns')

# Port for Prometheus to scrape
METRICS_PORT = 8000


def init_db():
    """
    Ensure the metrics SQLite database and table exist.
    """
    conn = sqlite3.connect(METRICS_DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            timestamp TEXT PRIMARY KEY,
            equity    REAL,
            sharpe    REAL,
            drawdown  REAL
        )
    """
    )
    conn.commit()
    conn.close()
    logger.info(f"Metrics DB initialized at {METRICS_DB_FILE}")


def collect_and_push():
    """
    Collect equity curve metrics, update Prometheus gauges, and persist to SQLite.
    """
    # Timestamp for this snapshot
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Fetch equity curve
    eq_curve = get_equity_curve()
    if not eq_curve:
        logger.warning("Empty equity curve; skipping metrics collection.")
        return

    # Current equity
    current_equity = eq_curve[-1]
    EQUITY_GAUGE.set(current_equity)

    # Compute returns, Sharpe, drawdown if possible
    sr_val = None
    dd_val = None
    if len(eq_curve) >= 2:
        eq = np.array(eq_curve)
        returns = np.diff(eq) / eq[:-1]
        sr_val = sharpe_ratio(returns)
        SHARPE_GAUGE.set(sr_val)
        dd_metrics = compute_drawdowns(eq)
        dd_val = dd_metrics['max_drawdown']
        DRAWDOWN_GAUGE.set(dd_val)

    # Persist to SQLite
    try:
        conn = sqlite3.connect(METRICS_DB_FILE)
        conn.execute(
            "INSERT OR REPLACE INTO metrics(timestamp, equity, sharpe, drawdown) VALUES (?, ?, ?, ?)",
            (ts, current_equity, sr_val, dd_val)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error writing metrics to DB: {e}")


def main():
    init_db()
    start_http_server(METRICS_PORT)
    logger.info(f"Prometheus metrics server listening on :{METRICS_PORT}/metrics")

    while True:
        try:
            collect_and_push()
        except Exception as e:
            logger.error(f"Unhandled error in metrics collection: {e}")
        time.sleep(30)


if __name__ == "__main__":
    main()
