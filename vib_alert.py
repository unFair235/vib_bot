#!/usr/bin/env python3
import websocket
import json
import time
from datetime import datetime
import csv
import os
import requests
import logging

# ========================
# CONFIG SECTION
# ========================

# Define your project directory (adjust if needed)
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

# Binance WebSocket URL for VIB/USDT trades
SOCKET_URL = "wss://stream.binance.com:9443/ws/vibusdt@trade"

# CSV log file path (now absolute, pointing to your project directory)
CSV_FILE = os.path.join(BASE_DIR, "vib_trades_log.csv")

# Telegram Bot Config
TELEGRAM_TOKEN = "7636229600:AAESoUoIB6nIcUHxme43x8byKhX1sok5zPk"
CHAT_ID = 531265494

# Minimum VIB quantity for big trade alerts
BIG_TRADE_THRESHOLD = 100000

# For exponential backoff on reconnects
reconnect_delay = 5
max_delay = 300  # 5 minutes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("vib_alert")

# ========================
# HELPER FUNCTIONS
# ========================

def send_telegram_alert(text):
    """
    Sends a message to your specified Telegram chat.
    Requires: requests library (pip install requests).
    """
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.warning("Telegram token or chat ID not set. Skipping Telegram alert.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": text
    }
    try:
        resp = requests.post(url, data=data, timeout=5)
        if resp.status_code != 200:
            logger.error(f"Telegram Error: {resp.text}")
    except Exception as e:
        logger.error(f"Telegram Exception: {e}")

def write_to_csv(data):
    """
    Appends trade data to CSV.
    Columns: [LocalTime, TradeID, Side, Price, Quantity, BuyerOrderID, SellerOrderID, TradeTime]
    """
    file_exists = os.path.isfile(CSV_FILE)
    
    with open(CSV_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # If file is brand new, write a header
        if not file_exists:
            writer.writerow([
                "LocalTime",
                "TradeID",
                "Side",
                "Price",
                "Quantity",
                "BuyerOrderID",
                "SellerOrderID",
                "TradeTime"
            ])
        
        writer.writerow(data)

# ========================
# WEBSOCKET CALLBACKS
# ========================

def on_message(ws, message):
    data = json.loads(message)

    # Extract fields
    trade_id = data.get("t")
    price = float(data.get("p"))
    qty = float(data.get("q"))
    buyer_order_id = data.get("b")
    seller_order_id = data.get("a")
    trade_time = data.get("T")   # in ms
    is_buyer_maker = data.get("m")

    # Convert times
    local_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trade_time_str = datetime.fromtimestamp(trade_time / 1000).strftime("%Y-%m-%d %H:%M:%S")

    # Determine side
    side = "BUY" if not is_buyer_maker else "SELL"

    # Log all trades
    logger.info(f"Trade - ID: {trade_id} | Side: {side} | Price: {price} | Qty: {qty}")

    # Write to CSV
    write_to_csv([
        local_time_str,
        trade_id,
        side,
        price,
        qty,
        buyer_order_id,
        seller_order_id,
        trade_time_str
    ])

    # Check for big trades
    if qty >= BIG_TRADE_THRESHOLD:
        total_value = price * qty
        alert_text = (
            f"🚨 BIG TRADE ALERT 🚨\n"
            f"Trade ID: {trade_id}\n"
            f"Side: {side}\n"
            f"Price: {price} USDT\n"
            f"Amount: {qty} VIB\n"
            f"Buyer Order ID: {buyer_order_id}\n"
            f"Seller Order ID: {seller_order_id}\n"
            f"Trade Time: {trade_time_str}\n"
            f"Total Value: {total_value:.2f} USDT\n"
        )
        # Log big trades
        logger.info(alert_text)
        # Send Telegram message
        send_telegram_alert(alert_text)

def on_error(ws, error):
    logger.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    """
    Called when the WebSocket connection closes. We'll attempt a reconnect with a delay.
    """
    global reconnect_delay
    logger.warning("WebSocket Closed.")
    logger.warning(f"Reconnecting in {reconnect_delay} seconds...")
    time.sleep(reconnect_delay)

    # Exponential backoff
    reconnect_delay = min(reconnect_delay * 2, max_delay)
    run_websocket()

def on_open(ws):
    """
    Called once the WebSocket connection is opened.
    We reset the reconnect delay to the default (5s).
    """
    global reconnect_delay
    reconnect_delay = 5
    logger.info("Connection Opened... Listening for Big Trades 🧐")

def run_websocket():
    ws = websocket.WebSocketApp(
        SOCKET_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    # Keep the connection alive with ping/pong
    ws.run_forever(ping_interval=20, ping_timeout=10)

# ========================
# MAIN LOOP
# ========================

if __name__ == "__main__":
    while True:
        try:
            run_websocket()
        except Exception as e:
            logger.error(f"Unhandled error: {e}")
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)