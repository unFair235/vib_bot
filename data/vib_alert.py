#!/usr/bin/env python3
import websocket
import json
import time
from datetime import datetime
import os
import requests
import logging
import sqlite3

BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"
SOCKET_URL = "wss://stream.binance.com:9443/ws/vibusdt@trade"
DB_FILE = os.path.join(BASE_DIR, "trades.db")
TELEGRAM_TOKEN = "7636229600:AAESoUoIB6nIcUHxme43x8byKhX1sok5zPk"
CHAT_ID = 531265494
BIG_TRADE_THRESHOLD = 100000
reconnect_delay = 5
max_delay = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("vib_alert")

def init_db():
    """Initialize the SQLite database for trades."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            local_time TEXT,
            trade_id INTEGER,
            side TEXT,
            price REAL,
            quantity REAL,
            buyer_order_id INTEGER,
            seller_order_id INTEGER,
            trade_time TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Trades database initialized and table ensured.")

def insert_trade(local_time, trade_id, side, price, quantity, buyer_order_id, seller_order_id, trade_time):
    """Insert a trade record into the trades table."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades (
                local_time, trade_id, side, price, quantity, buyer_order_id, seller_order_id, trade_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (local_time, trade_id, side, price, quantity, buyer_order_id, seller_order_id, trade_time))
        conn.commit()
    except Exception as e:
        logger.error(f"Error inserting trade: {e}")
    finally:
        conn.close()

init_db()

def send_telegram_alert(message, retries=3):
    """Send Telegram alert with retry mechanism."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    for attempt in range(retries):
        try:
            resp = requests.post(url, data=data, timeout=5)
            if resp.status_code == 200:
                return
            else:
                logger.error(f"Telegram Error: {resp.text}")
        except Exception as e:
            logger.error(f"Telegram Exception on attempt {attempt+1}: {e}")
        time.sleep(2)

def on_message(ws, message):
    try:
        data = json.loads(message)
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return
    trade_id = data.get("t")
    try:
        price = float(data.get("p"))
        qty = float(data.get("q"))
    except Exception as e:
        logger.error(f"Error converting price/qty: {e}")
        return
    buyer_order_id = data.get("b")
    seller_order_id = data.get("a")
    trade_time_ms = data.get("T")
    is_buyer_maker = data.get("m")
    local_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        trade_time_str = datetime.fromtimestamp(trade_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Error converting trade time: {e}")
        trade_time_str = "N/A"
    side = "BUY" if not is_buyer_maker else "SELL"
    logger.info(f"Trade - ID: {trade_id} | Side: {side} | Price: {price} | Qty: {qty}")
    insert_trade(local_time_str, trade_id, side, price, qty, buyer_order_id, seller_order_id, trade_time_str)
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
        logger.info(alert_text)
        send_telegram_alert(alert_text)

def on_error(ws, error):
    logger.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    global reconnect_delay
    logger.warning("WebSocket Closed.")
    logger.warning(f"Reconnecting in {reconnect_delay} seconds...")
    time.sleep(reconnect_delay)
    reconnect_delay = min(reconnect_delay * 2, max_delay)
    run_websocket()

def on_open(ws):
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
    ws.run_forever(ping_interval=20, ping_timeout=10)

if __name__ == "__main__":
    while True:
        try:
            run_websocket()
        except Exception as e:
            logger.error(f"Unhandled error: {e}")
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)