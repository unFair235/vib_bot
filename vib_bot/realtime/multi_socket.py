#!/usr/bin/env python3
"""
realtime/multi_socket.py

Spawns the combined trade‐feed websocket plus per‐symbol orderbook websockets,
reconnecting automatically on any crash.
"""
import threading
import time
import os
import logging
from logging.handlers import RotatingFileHandler

from vib_bot.config import BASE_DIR, SYMBOLS
from vib_bot.data.vib_alert import run_ws as run_trade_ws
from vib_bot.data.vib_orderbook import run_symbol_ws

# ——— Configuration ———
LOG_FILE        = os.path.join(BASE_DIR, "multi_socket.log")
RECONNECT_DELAY = int(os.getenv("VIB_BOT_RECONNECT_DELAY", "5"))  # seconds

# ——— Logging Setup ———
logger = logging.getLogger("multi_socket")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
rot_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
rot_handler.setFormatter(fmt)
logger.addHandler(rot_handler)
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def trade_loop():
    """Continuously run the combined trade websocket, reconnecting on failure."""
    while True:
        try:
            logger.info("▶️  Starting combined trade socket")
            run_trade_ws()
        except Exception:
            logger.exception(
                "Combined trade socket crashed; reconnecting in %d seconds", RECONNECT_DELAY
            )
        time.sleep(RECONNECT_DELAY)


def orderbook_loop(symbol: str):
    """Continuously run the orderbook websocket for `symbol`, reconnecting on failure."""
    while True:
        try:
            logger.info(f"▶️  Starting orderbook socket for {symbol}")
            run_symbol_ws(symbol)
        except Exception:
            logger.exception(
                "Orderbook socket for %s crashed; reconnecting in %d seconds",
                symbol, RECONNECT_DELAY
            )
        time.sleep(RECONNECT_DELAY)


def main():
    # 1) start the combined trade‐feed thread
    logger.info("🧵 Spawning combined trade websocket thread")
    t_trade = threading.Thread(target=trade_loop, name="TradeThread", daemon=True)
    t_trade.start()

    # 2) start one orderbook thread per symbol
    logger.info("🧵 Spawning orderbook websocket threads for symbols: %s", ", ".join(SYMBOLS))
    ob_threads = []
    for sym in SYMBOLS:
        t = threading.Thread(
            target=orderbook_loop,
            args=(sym,),
            name=f"{sym}-OBThread",
            daemon=True
        )
        t.start()
        ob_threads.append(t)

    # keep main alive
    t_trade.join()
    for t in ob_threads:
        t.join()


if __name__ == "__main__":
    main()