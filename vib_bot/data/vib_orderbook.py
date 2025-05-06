#!/usr/bin/env python3
import os
import time
import json
import sqlite3
import logging
import threading
import websocket

from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from vib_bot.config import BASE_DIR, ORDERBOOK_DB_FILE, SYMBOLS

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────
DB_FILE           = ORDERBOOK_DB_FILE
LOG_FILE          = os.path.join(BASE_DIR, "vib_orderbook.log")
DEPTH_CHANNEL_FMT = "{symbol_lower}@depth5"
WS_BASE           = "wss://stream.binance.com:9443/ws/"

RECONNECT_DELAY = 5  # seconds

# ────────────────────────────────────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("vib_orderbook")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Console
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

# Rotating file
fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024**2, backupCount=3)
fh.setFormatter(fmt)
logger.addHandler(fh)


# ────────────────────────────────────────────────────────────────────────────────
# Database initialization
# ────────────────────────────────────────────────────────────────────────────────
def init_db():
    """Ensure orderbook_data table exists with a symbol column."""
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orderbook_data (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol    TEXT,
            best_bid  REAL,
            best_ask  REAL,
            spread    REAL
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Orderbook database initialized.")


init_db()


# ────────────────────────────────────────────────────────────────────────────────
# Insert snapshot
# ────────────────────────────────────────────────────────────────────────────────
def insert_snapshot(symbol: str, best_bid: float, best_ask: float, spread: float):
    ts = datetime.now(timezone.utc).isoformat()
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            "INSERT INTO orderbook_data (timestamp, symbol, best_bid, best_ask, spread)"
            " VALUES (?, ?, ?, ?, ?)",
            (ts, symbol, best_bid, best_ask, spread)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"[{symbol}] DB insert error: {e}")
    finally:
        conn.close()


# ────────────────────────────────────────────────────────────────────────────────
# WebSocket callbacks factory
# ────────────────────────────────────────────────────────────────────────────────
def make_callbacks(symbol: str):
    symbol_upper = symbol
    def on_message(ws, message: str):
        try:
            data = json.loads(message)
            bids = data.get("bids", [])
            asks = data.get("asks", [])
        except Exception as e:
            return logger.error(f"[{symbol_upper}] JSON parse error: {e}")

        if not bids or not asks:
            return

        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread   = best_ask - best_bid
            logger.info(f"[{symbol_upper}] Bid {best_bid:.4f}, Ask {best_ask:.4f}, Spread {spread:.4f}")
            insert_snapshot(symbol_upper, best_bid, best_ask, spread)
        except Exception as e:
            logger.error(f"[{symbol_upper}] processing error: {e}")

    def on_error(ws, error):
        logger.error(f"[{symbol_upper}] websocket error: {error}")

    def on_close(ws, code, reason):
        logger.warning(f"[{symbol_upper}] websocket closed: {code}/{reason}")

    def on_open(ws):
        logger.info(f"[{symbol_upper}] websocket opened")

    return on_open, on_message, on_error, on_close


# ────────────────────────────────────────────────────────────────────────────────
# Runner for a single symbol
# ────────────────────────────────────────────────────────────────────────────────
def run_symbol_ws(symbol: str):
    symbol_lower = symbol.lower()
    channel      = DEPTH_CHANNEL_FMT.format(symbol_lower=symbol_lower)
    url          = WS_BASE + channel

    on_open, on_message, on_error, on_close = make_callbacks(symbol)

    while True:
        try:
            ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.exception(f"[{symbol}] fatal error, reconnecting in {RECONNECT_DELAY}s", e)
        time.sleep(RECONNECT_DELAY)


# ────────────────────────────────────────────────────────────────────────────────
# Main entrypoint — spawn one thread per symbol
# ────────────────────────────────────────────────────────────────────────────────
def main():
    if not SYMBOLS:
        logger.error("No SYMBOLS configured; exiting.")
        return

    threads = []
    for sym in SYMBOLS:
        t = threading.Thread(
            target=run_symbol_ws,
            name=f"Orderbook-{sym}",
            args=(sym,),
            daemon=True
        )
        t.start()
        threads.append(t)
        logger.info(f"Started orderbook thread for {sym}")

    # keep alive
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()