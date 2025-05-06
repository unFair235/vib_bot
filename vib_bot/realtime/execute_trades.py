#!/usr/bin/env python3
"""
realtime/execute_trades.py

Send orders to Binance with a basic slippage guard and optional OCO for stop‑loss/take‑profit.
"""
import os
import time
import hmac
import hashlib
import logging
import requests
from urllib.parse import urlencode

from vib_bot.config import SLIPPAGE_TOLERANCE

logger = logging.getLogger("execute_trades")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


def _sign(params: dict) -> str:
    """HMAC‑SHA256 sign a Binance request."""
    api_secret = os.getenv("BINANCE_API_SECRET")
    query = urlencode(params)
    return hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def place_order(
    symbol: str,
    side: str,
    quantity: float,
    stop_loss: float = None,
    take_profit: float = None
) -> dict:
    """
    Place a market order on Binance with an optional OCO for stop‑loss / take‑profit.
    Returns a dict with 'order' and (if placed) 'oco' responses.
    Requires BINANCE_API_KEY and BINANCE_API_SECRET in the environment.
    """
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    base_url = os.getenv("BINANCE_API_BASE", "https://api.binance.com")

    if not api_key or not api_secret:
        raise RuntimeError("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")

    # 1) Fetch current price for slippage check
    try:
        resp = requests.get(f"{base_url}/api/v3/ticker/price", params={"symbol": symbol}, timeout=5)
        resp.raise_for_status()
        current_price = float(resp.json()["price"])
    except Exception as e:
        logger.error(f"Failed to fetch ticker price: {e}")
        raise

    # 2) Assemble market order parameters
    timestamp = int(time.time() * 1000)
    order_params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": "MARKET",
        "quantity": f"{quantity:.8f}",
        "timestamp": timestamp
    }
    order_params["signature"] = _sign(order_params)
    headers = {"X-MBX-APIKEY": api_key}

    # 3) Submit the market order
    try:
        r = requests.post(f"{base_url}/api/v3/order", params=order_params, headers=headers, timeout=5)
        r.raise_for_status()
        order_result = r.json()
        fills = order_result.get("fills", [])
        avg_price = (
            sum(float(f["price"]) * float(f["qty"]) for f in fills) /
            sum(float(f["qty"]) for f in fills)
        ) if fills else current_price
    except Exception as e:
        logger.error(f"Market order failed: {e}")
        raise

    # 4) Slippage guard
    limit_price = (
        current_price * (1 + SLIPPAGE_TOLERANCE)
        if side.upper() == "BUY"
        else current_price * (1 - SLIPPAGE_TOLERANCE)
    )
    if (side.upper() == "BUY" and avg_price > limit_price) or \
       (side.upper() == "SELL" and avg_price < limit_price):
        logger.error(f"Slippage violation: executed @ {avg_price:.8f}, limit was {limit_price:.8f}")
    else:
        logger.info(f"Executed {side.upper()} {quantity} {symbol} @ {avg_price:.8f}")

    # 5) Optionally place an OCO order for stop‑loss / take‑profit
    oco_result = None
    if stop_loss is not None and take_profit is not None and side.upper() == "BUY":
        oco_timestamp = int(time.time() * 1000)
        oco_params = {
            "symbol": symbol,
            "side": "SELL",
            "quantity": f"{quantity:.8f}",
            "price": f"{take_profit:.8f}",
            "stopPrice": f"{stop_loss:.8f}",
            "stopLimitPrice": f"{(stop_loss * 0.995):.8f}",
            "stopLimitTimeInForce": "GTC",
            "timestamp": oco_timestamp
        }
        oco_params["signature"] = _sign(oco_params)
        try:
            oco_resp = requests.post(f"{base_url}/api/v3/order/oco", params=oco_params, headers=headers, timeout=5)
            oco_resp.raise_for_status()
            oco_result = oco_resp.json()
            logger.info("OCO stop‑loss/take‑profit order placed")
        except Exception as e:
            logger.error(f"OCO order failed: {e}")

    return {"order": order_result, "oco": oco_result}