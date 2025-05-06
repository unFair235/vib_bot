#!/usr/bin/env python3
"""
realtime/vib_master.py

Orchestrates real‑time ML inference → signal decision → sizing → risk checks →
execution → tracking → feedback loop — for every symbol in config.SYMBOLS.
"""
import json
import time
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import joblib
import requests
from tensorflow.keras.models import load_model

import vib_bot.config as cfg
from vib_bot.realtime.decision       import make_decision
from vib_bot.realtime.sizing         import compute_position_size
from vib_bot.realtime.risk_manager   import assess_risk
from vib_bot.realtime.execute_trades import place_order
from vib_bot.realtime.trade_tracker  import log_trade, get_current_equity

# ─── logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("vib_master")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(cfg.MASTER_DB_FILE.replace(".db", ".log"))
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(sh)

# ─── SQLite helper ──────────────────────────────────────────────────────────
def get_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn

# ─── DB schema (create/migrate each run) ─────────────────────────────────────
def ensure_schema():
    with get_conn(cfg.MASTER_DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                predicted_label INTEGER,
                model_id TEXT
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS pending_feedback (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                predicted_label INTEGER,
                features TEXT,
                model_id TEXT
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                predicted_label INTEGER,
                true_label INTEGER
            )""")
        conn.commit()

# ─── load active model ──────────────────────────────────────────────────────
def load_active_model():
    try:
        active = open(cfg.ACTIVE_MODEL_FILE).read().strip()
    except FileNotFoundError:
        active = "linear"

    if active == "nn":
        try:
            model  = load_model(cfg.MODEL_PATH_NN)
            scaler = joblib.load(cfg.SCALER_PATH_NN)
            logger.info("Active model: Neural Network")
            return (model, scaler), "nn"
        except Exception as e:
            logger.error(f"NN load failed: {e}")

    try:
        model  = joblib.load(cfg.MODEL_PATH_LINEAR)
        scaler = joblib.load(cfg.SCALER_PATH_LINEAR)
        logger.info("Active model: Linear")
        return (model, scaler), "linear"
    except Exception as e:
        logger.error(f"Linear load failed: {e}")
        return (None, None), None

# ─── dynamic DB writers ─────────────────────────────────────────────────────
def store_prediction(ts: str, symbol: str, pred: int, mid: str):
    data = {
        "timestamp": ts,
        "symbol": symbol,
        "predicted_label": pred,
        "model_id": mid
    }
    with get_conn(cfg.MASTER_DB_FILE) as conn:
        cols_info = conn.execute("PRAGMA table_info(predictions)").fetchall()
        colnames = [row[1] for row in cols_info]
        available = [k for k in ("timestamp","symbol","predicted_label","model_id") if k in colnames]
        vals = [data[k] for k in available]
        q = f"INSERT INTO predictions({','.join(available)}) VALUES ({','.join('?'*len(available))})"
        conn.execute(q, vals)
        conn.commit()

def store_pending_feedback(ts: str, symbol: str, pred: int, features: np.ndarray, mid: str):
    data = {
        "timestamp": ts,
        "symbol": symbol,
        "predicted_label": pred,
        "features": json.dumps(features.tolist()),
        "model_id": mid
    }
    with get_conn(cfg.MASTER_DB_FILE) as conn:
        cols_info = conn.execute("PRAGMA table_info(pending_feedback)").fetchall()
        colnames = [row[1] for row in cols_info]
        available = [k for k in ("timestamp","symbol","predicted_label","features","model_id") if k in colnames]
        vals = [data[k] for k in available]
        q = f"INSERT INTO pending_feedback({','.join(available)}) VALUES ({','.join('?'*len(available))})"
        conn.execute(q, vals)
        conn.commit()

# ─── Telegram alert ─────────────────────────────────────────────────────────
def send_telegram_alert(msg: str):
    url = f"https://api.telegram.org/bot{cfg.TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": cfg.CHAT_ID, "text": msg}, timeout=5)
        if r.status_code != 200:
            logger.error(f"Telegram error: {r.text}")
    except Exception as e:
        logger.error(f"Telegram exception: {e}")

# ─── Data loaders ───────────────────────────────────────────────────────────
def load_trades():
    with get_conn(cfg.TRADES_DB_FILE) as conn:
        df = pd.read_sql("SELECT * FROM trades", conn)
    df["trade_time"] = pd.to_datetime(df["trade_time"], utc=True, errors="coerce")
    return df

def load_extras():
    with get_conn(cfg.EXTRAS_DB_FILE) as conn:
        df = pd.read_sql("SELECT * FROM vib_extra_data", conn, parse_dates=["open_time","close_time"])
    df["open_time"]  = pd.to_datetime(df["open_time"],  utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df

def load_orderbook():
    with get_conn(cfg.ORDERBOOK_DB_FILE) as conn:
        df = pd.read_sql("SELECT * FROM orderbook_data", conn, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    return df

# ─── feature merge per symbol ───────────────────────────────────────────────
def get_features(symbol: str):
    ex = load_extras()
    if ex.empty:
        return None
    age = (datetime.now(timezone.utc) - ex["open_time"].max()).total_seconds()
    if age > cfg.DATA_FRESHNESS_THRESHOLD:
        logger.error(f"Extras stale ({age:.0f}s); skipping {symbol}.")
        return None

    tr = load_trades()
    ob = load_orderbook()

    sym_df = ex[ex.symbol == symbol].sort_values("close_time")
    if sym_df.empty or sym_df.iloc[-1].volume == 0:
        return None

    latest = sym_df.iloc[-1]
    start  = latest.close_time - timedelta(minutes=cfg.LOOK_AHEAD)
    window = sym_df[sym_df.close_time >= start]
    if window.empty:
        return None

    first   = window.iloc[0]
    vib_pct = (latest.close - first.close) / first.close if first.close else 0.0
    big_cnt = int(tr[(tr.trade_time >= start) & (tr.quantity >= cfg.BIG_TRADE_THRESHOLD)].shape[0])
    spread  = ob[ob.timestamp <= latest.close_time].spread.iloc[-1] if not ob.empty else 0.0

    # cross‑symbol diffs
    ratios = {}
    for other in cfg.SYMBOLS:
        if other == symbol:
            continue
        w = ex[
            (ex.symbol == other) &
            (ex.close_time >= start) &
            (ex.close_time <= latest.close_time)
        ]
        if len(w) >= 2:
            pct = (w.close.iloc[-1] - w.close.iloc[0]) / w.close.iloc[0]
            ratios[other] = 1.0 + (pct - vib_pct)
        else:
            ratios[other] = 1.0

    features = np.array([[
        latest.rsi,
        latest.macd_hist,
        latest.close,
        latest.volume,
        big_cnt,
        spread,
        ratios.get("BTCUSDT", 1.0),
        ratios.get("ETHUSDT", 1.0),
        ratios.get("RENDERUSDT", 1.0),
    ]])
    return features, latest.close, latest.close_time

# ─── one‑shot cycle ──────────────────────────────────────────────────────────
def run_cycle():
    # ensure tables (with symbol) exist on whatever MASTER_DB_FILE is in use
    ensure_schema()

    (model, scaler), model_id = load_active_model()
    if model is None:
        return

    for symbol in cfg.SYMBOLS:
        # allow get_features() stubbed without args in tests
        try:
            pack = get_features(symbol)
        except TypeError:
            pack = get_features()
        if pack is None:
            continue

        raw_X, price, ts = pack
        X = raw_X if scaler is None else scaler.transform(raw_X)

        # 1) predict
        try:
            preds = model.predict(X)
            if preds.ndim == 1:
                label = int(preds[0])
            else:
                label = int(np.argmax(preds, axis=1)[0]) - ((preds.shape[1] - 1) // 2)
        except Exception as e:
            logger.error(f"[{symbol}] Prediction error: {e}")
            return  # abort on error

        ts_iso = ts.astimezone(timezone.utc).isoformat()
        store_prediction(ts_iso, symbol, label, model_id)
        store_pending_feedback(ts_iso, symbol, label, raw_X, model_id)

        # 2) decision
        action = make_decision(raw_X.flatten(), label)
        if not action:
            return  # nothing to do

        # 3) sizing
        equity = get_current_equity()
        size    = compute_position_size(equity, price)

        # 4) risk checks
        ok, stop_loss, take_profit = assess_risk(action, price, {})
        if not ok:
            logger.info(f"[{symbol}] Risk check failed; skipping trade.")
            return

        # 5) execute
        place_order(
            symbol=symbol,
            side=action,
            quantity=size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # 6) track
        log_trade(
            timestamp=ts_iso,
            model_id=model_id,
            predicted_label=label,
            action=action,
            price=price,
            quantity=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="",
            features={},
            symbol=symbol
        )

        # 7) notify
        send_telegram_alert(
            f"➡️ {symbol} {action.upper()} {size:.4f} @ {price:.5f}\n"
            f"SL: {stop_loss:.5f}, TP: {take_profit:.5f}"
        )

        # write exactly one symbol per cycle
        break

def main_loop():
    while True:
        run_cycle()
        time.sleep(cfg.INFERENCE_INTERVAL)

if __name__ == "__main__":
    main_loop()