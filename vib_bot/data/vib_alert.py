#!/usr/bin/env python3
import os
import time
import json
import logging
import sqlite3
import requests
import websocket

from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from vib_bot.config import (
    BASE_DIR,
    TRADES_DB_FILE,
    TELEGRAM_TOKEN,
    CHAT_ID,
    BIG_TRADE_THRESHOLD,
    SYMBOLS,
)

# ────────────────────────────────────────────────────────────────────────────────
# Build combined trade‐stream URL for all SYMBOLS
# ────────────────────────────────────────────────────────────────────────────────
if not SYMBOLS:
    raise RuntimeError("No SYMBOLS configured; cannot subscribe to any trade streams.")

stream_paths = [f"{sym.lower()}@trade" for sym in SYMBOLS]
SOCKET_URL = f"wss://stream.binance.com:9443/stream?streams={'/'.join(stream_paths)}"

DB_FILE = TRADES_DB_FILE

# ────────────────────────────────────────────────────────────────────────────────
# Logging setup with rotation
# ────────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("vib_alert")
logger.setLevel(logging.INFO)

log_path = os.path.join(BASE_DIR, "vib_alert.log")
rot_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024**2, backupCount=3)
rot_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(rot_handler)

console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console)


# ────────────────────────────────────────────────────────────────────────────────
# DB initialization (adds 'symbol' column)
# ────────────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol           TEXT,
            local_time       TEXT,
            trade_id         INTEGER,
            side             TEXT,
            price            REAL,
            quantity         REAL,
            buyer_order_id   INTEGER,
            seller_order_id  INTEGER,
            trade_time       TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Trades DB initialized (with symbol column).")

init_db()


# ────────────────────────────────────────────────────────────────────────────────
# Insert a trade record
# ────────────────────────────────────────────────────────────────────────────────
def insert_trade(
    symbol: str,
    local_time: str,
    trade_id: int,
    side: str,
    price: float,
    qty: float,
    buyer_id: int,
    seller_id: int,
    trade_time: str
):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            """
            INSERT INTO trades(
                symbol, local_time, trade_id, side,
                price, quantity, buyer_order_id,
                seller_order_id, trade_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, local_time, trade_id, side, price, qty, buyer_id, seller_id, trade_time)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Insert trade error: {e}")
    finally:
        conn.close()


# ────────────────────────────────────────────────────────────────────────────────
# Telegram alert for big trades
# ────────────────────────────────────────────────────────────────────────────────
def send_telegram_alert(msg: str, retries: int = 3):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg}
    for i in range(retries):
        try:
            r = requests.post(url, data=data, timeout=5)
            if r.status_code == 200:
                return
            logger.error(f"Telegram error ({r.status_code}): {r.text}")
        except Exception as e:
            logger.error(f"Telegram exception on try {i+1}: {e}")
        time.sleep(2)


# ────────────────────────────────────────────────────────────────────────────────
# WebSocket callbacks
# ────────────────────────────────────────────────────────────────────────────────
def on_message(ws, message: str):
    """
    Binance combined stream sends messages of the form:
      {"stream":"btcusdt@trade","data":{...}}
    """
    try:
        wrapper = json.loads(message)
        data    = wrapper["data"]
        # Extract the symbol from the wrapper
        stream_name = wrapper.get("stream", "")
        symbol      = stream_name.split("@")[0].upper()
        t_id        = data["t"]
        price       = float(data["p"])
        qty         = float(data["q"])
    except Exception as e:
        return logger.error(f"Parse error: {e}")

    # Side: m == True means a sell, otherwise buy
    side = "SELL" if data.get("m", False) else "BUY"
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        trade_ts = datetime.fromtimestamp(data["T"] / 1000, timezone.utc) \
                          .strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        trade_ts = ""

    logger.info(f"[{symbol}] Trade {t_id} {side} {price}@{qty}")
    insert_trade(symbol, now, t_id, side, price, qty, data["b"], data["a"], trade_ts)

    # Big‐trade alert
    if qty >= BIG_TRADE_THRESHOLD:
        total = price * qty
        alert = (
            f"🚨 BIG TRADE 🚨\n"
            f"Symbol: {symbol}\n"
            f"ID:     {t_id}\n"
            f"Side:   {side}\n"
            f"Price:  {price}\n"
            f"Qty:    {qty}\n"
            f"Time:   {trade_ts}\n"
            f"Value:  {total:.2f} USDT"
        )
        logger.info(alert.replace("\n", " | "))
        send_telegram_alert(alert)


def on_error(ws, err):
    logger.error(f"WebSocket error: {err}")


def on_close(ws, code, reason):
    logger.warning(f"WS closed {code}/{reason}; reconnecting in 5s")
    time.sleep(5)


def on_open(ws):
    logger.info("WebSocket connection opened.")


# ────────────────────────────────────────────────────────────────────────────────
# Runner (reconnect loop)
# ────────────────────────────────────────────────────────────────────────────────
def run_ws():
    ws = websocket.WebSocketApp(
        SOCKET_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever(ping_interval=20, ping_timeout=10)


if __name__ == "__main__":
    while True:
        try:
            run_ws()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        logger.info("Reconnecting trade stream in 5s…")
        time.sleep(5)