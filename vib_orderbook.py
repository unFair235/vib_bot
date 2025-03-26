#!/usr/bin/env python3
import websocket
import json
import time
import csv
import os
from datetime import datetime
import logging

# Define the base directory for your project
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

# Absolute CSV file path for storing order book snapshots
CSV_FILE = os.path.join(BASE_DIR, "orderbook_data.csv")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("vib_orderbook")

# Binance WebSocket URL for VIB/USDT depth data
ORDER_BOOK_URL = "wss://stream.binance.com:9443/ws/vibusdt@depth5"

def log_orderbook(best_bid, best_ask, spread):
    try:
        logger.info("Inside log_orderbook: preparing to write data.")
        file_exists = os.path.isfile(CSV_FILE)
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "best_bid", "best_ask", "spread"])
                logger.info("Header written to CSV.")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, best_bid, best_ask, spread])
            f.flush()
            logger.info("Data row successfully written to CSV.")
    except Exception as e:
        logger.error(f"Error in log_orderbook: {e}")

def on_message(ws, message):
    logger.info(f"Raw message received: {message}")
    try:
        data = json.loads(message)
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return
    
    bids = data.get('bids', [])
    asks = data.get('asks', [])
    logger.info(f"bids count: {len(bids)}, asks count: {len(asks)}")
    if bids and asks:
        try:
            logger.info("About to call log_orderbook")
            best_bid = round(float(bids[0][0]), 4)
            best_ask = round(float(asks[0][0]), 4)
            spread = best_ask - best_bid
            log_orderbook(best_bid, best_ask, spread)
            logger.info(f"Logged at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Bid: {best_bid}, Ask: {best_ask}, Spread: {spread}")
        except Exception as e:
            logger.error(f"Error processing order book data: {e}")

def on_error(ws, error):
    logger.error(f"Order book error: {error}")

def on_close(ws, code, reason):
    logger.warning("Order book feed closed. Reconnecting in 5s...")
    time.sleep(5)
    run_orderbook()

def on_open(ws):
    logger.info("Order book WebSocket connected.")

def run_orderbook():
    ws = websocket.WebSocketApp(
        ORDER_BOOK_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever(ping_interval=20, ping_timeout=10)

if __name__ == "__main__":
    run_orderbook()