#!/usr/bin/env python3
import websocket
import json
import time
from datetime import datetime
import os
import sqlite3
import logging

BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"
ORDER_BOOK_URL = "wss://stream.binance.com:9443/ws/vibusdt@depth5"
DB_FILE = os.path.join(BASE_DIR, "orderbook.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("vib_orderbook")

def init_db():
    """Initialize the SQLite database for orderbook snapshots."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orderbook_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            best_bid REAL,
            best_ask REAL,
            spread REAL
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Orderbook database initialized (orderbook_data table ensured).")

init_db()

def insert_orderbook(best_bid, best_ask, spread):
    """Insert an orderbook snapshot into the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("""
            INSERT INTO orderbook_data (timestamp, best_bid, best_ask, spread)
            VALUES (?, ?, ?, ?)
        """, (timestamp, best_bid, best_ask, spread))
        conn.commit()
    except Exception as e:
        logger.error(f"Error inserting orderbook snapshot: {e}")
    finally:
        conn.close()

def on_message(ws, message):
    try:
        data = json.loads(message)
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return
    bids = data.get('bids', [])
    asks = data.get('asks', [])
    logger.info(f"Received orderbook snapshot: {len(bids)} bids, {len(asks)} asks")
    if bids and asks:
        try:
            best_bid = round(float(bids[0][0]), 4)
            best_ask = round(float(asks[0][0]), 4)
            spread = best_ask - best_bid
            logger.info(f"Orderbook: Bid {best_bid}, Ask {best_ask}, Spread {spread:.4f}")
            insert_orderbook(best_bid, best_ask, spread)
        except Exception as e:
            logger.error(f"Error processing orderbook data: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.warning("Orderbook WebSocket Closed. Reconnecting in 5 seconds...")
    time.sleep(5)
    run_orderbook()

def on_open(ws):
    logger.info("Orderbook WebSocket connected. Listening for updates...")

def run_orderbook():
    ws = websocket.WebSocketApp(
        ORDER_BOOK_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever(ping_interval=20, ping_timeout=10)

if __name__ == "__main__":
    while True:
        try:
            run_orderbook()
        except Exception as e:
            logger.error(f"Unhandled error: {e}")
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)