#!/usr/bin/env python3
import threading
import logging
import vib_alert
import vib_orderbook

# Set up logging for this script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("multi_socket")

def run_trade_socket():
    try:
        logger.info("Starting trade socket (vib_alert).")
        vib_alert.run_websocket()  # This calls the trade feed from vib_alert.py
    except Exception as e:
        logger.error("Error in run_trade_socket: %s", e)

def run_orderbook_socket():
    try:
        logger.info("Starting orderbook socket (vib_orderbook).")
        vib_orderbook.run_orderbook()  # This calls the order book feed from vib_orderbook.py
    except Exception as e:
        logger.error("Error in run_orderbook_socket: %s", e)

if __name__ == "__main__":
    logger.info("Multi-socket script starting.")
    t1 = threading.Thread(target=run_trade_socket)
    t2 = threading.Thread(target=run_orderbook_socket)

    t1.start()
    t2.start()

    t1.join()
    t2.join()