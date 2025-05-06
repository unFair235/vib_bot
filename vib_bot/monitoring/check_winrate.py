#!/usr/bin/env python3
import argparse
import sqlite3
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

import vib_bot.config as cfg

DEFAULT_SYMBOL = cfg.SYMBOLS[0] if cfg.SYMBOLS else None


def ensure_symbol_columns():
    """
    Ensure the `symbol` column exists in predictions and feedback tables.
    If missing, add it and backfill with DEFAULT_SYMBOL.
    """
    conn = sqlite3.connect(cfg.MASTER_DB_FILE)
    cur = conn.cursor()
    for table in ("predictions", "feedback"):
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        if "symbol" not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN symbol TEXT")
            cur.execute(f"UPDATE {table} SET symbol = ?", (DEFAULT_SYMBOL,))
    conn.commit()
    conn.close()


def parse_args():
    """
    Parse CLI arguments, ignoring unknown flags (e.g., pytest options).
    """
    parser = argparse.ArgumentParser(
        description="Compute prediction winrate and report."
    )
    parser.add_argument(
        "--symbol", "-s",
        choices=cfg.SYMBOLS,
        help="Symbol to filter (e.g., BTCUSDT). If omitted, uses all symbols."
    )
    # parse_known_args so pytest's flags won't cause an error
    args, _ = parser.parse_known_args()
    return args


def main():
    # 1) Make sure every row in both tables has a symbol column
    ensure_symbol_columns()

    # 2) Parse --symbol if provided
    args = parse_args()
    symbol_filter = args.symbol

    # 3) Load both tables, now including `symbol`
    conn = sqlite3.connect(cfg.MASTER_DB_FILE)
    df_pred = pd.read_sql_query(
        "SELECT timestamp, predicted_label, symbol FROM predictions",
        conn,
        parse_dates=["timestamp"],
    )
    df_feed = pd.read_sql_query(
        "SELECT timestamp, true_label, symbol FROM feedback",
        conn,
        parse_dates=["timestamp"],
    )
    conn.close()

    # 4) Optionally filter by symbol
    if symbol_filter:
        df_pred = df_pred[df_pred.symbol == symbol_filter]
        df_feed = df_feed[df_feed.symbol == symbol_filter]

    # 5) Join on timestamp + symbol
    df = pd.merge(
        df_pred,
        df_feed,
        on=["timestamp", "symbol"],
        how="inner"
    )

    # 6) If no matches, still print zero‐line and exit
    total = len(df)
    if total == 0:
        print("Winrate (Accuracy): 0.00% over 0 samples")
        return

    # 7) Compute accuracy and print
    acc = accuracy_score(df["true_label"], df["predicted_label"]) * 100
    print(f"Winrate (Accuracy): {acc:.2f}% over {total} samples\n")

    # 8) Print a standard classification report
    print("Classification Report:")
    print(classification_report(df["true_label"], df["predicted_label"]))

    # 9) And show the first few rows for a sanity check
    print("\nSample predictions vs true labels:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()