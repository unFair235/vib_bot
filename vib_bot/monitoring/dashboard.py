#!/usr/bin/env python3
# monitoring/dashboard.py

from fastapi import FastAPI, Query, HTTPException
from vib_bot.config import MASTER_DB_FILE, SYMBOLS
import pandas as pd
import sqlite3
from sklearn.metrics import accuracy_score, classification_report

DEFAULT_SYMBOL = SYMBOLS[0] if SYMBOLS else "VIBUSDT"

def ensure_symbol_columns():
    """
    Ensure a `symbol` column exists in both predictions and feedback tables.
    If missing, add it and backfill with DEFAULT_SYMBOL.
    """
    conn = sqlite3.connect(MASTER_DB_FILE)
    cur = conn.cursor()
    for table in ("predictions", "feedback"):
        cur.execute(f"PRAGMA table_info({table})")
        cols = [info[1] for info in cur.fetchall()]
        if "symbol" not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN symbol TEXT")
            cur.execute(f"UPDATE {table} SET symbol = ?", (DEFAULT_SYMBOL,))
    conn.commit()
    conn.close()

# Ensure schema is up to date before serving any requests
ensure_symbol_columns()

app = FastAPI(title="Altcoins Monitoring Dashboard")

@app.get("/api/metrics")
def get_metrics(symbol: str = Query(..., description="Symbol from configured list")):
    # Validate symbol
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail=f"{symbol} not in configured symbol list")

    # Query predictions + feedback for that symbol
    with sqlite3.connect(MASTER_DB_FILE) as conn:
        df = pd.read_sql_query(
            """
            SELECT p.predicted_label AS predicted,
                   f.true_label      AS truth
            FROM predictions p
            JOIN feedback   f ON p.timestamp = f.timestamp
            WHERE p.symbol = ?
            """,
            conn,
            params=(symbol,)
        )

    if df.empty:
        return {"symbol": symbol, "message": "No data available"}

    acc = accuracy_score(df["truth"], df["predicted"])
    report = classification_report(df["truth"], df["predicted"], output_dict=True)
    return {"symbol": symbol, "accuracy": acc, "report": report}