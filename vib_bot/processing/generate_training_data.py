#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd

from vib_bot.config import BASE_DIR, LOOK_AHEAD, SYMBOLS
from vib_bot.processing.utils import (
    load_extras,
    load_trades,
    load_orderbook,
    bucket_label,
    compute_feature_windows,
)

TRAINING_DB = os.path.join(BASE_DIR, "training_data.db")

def build_dataset(
    extras: pd.DataFrame,
    trades: pd.DataFrame,
    orderbook: pd.DataFrame,
    look_ahead: int
) -> pd.DataFrame:
    """
    For each symbol in SYMBOLS:
      - aligns each candle with its future close (look_ahead minutes ahead)
      - buckets the % change into [-3..+3]
      - computes your windowed features
    Returns one big DataFrame with columns:
      timestamp, rsi, macd_hist, vib_close, volume,
      big_trades_count, orderbook_spread,
      diff_BTC, diff_ETH, diff_RNDR, label
    """
    all_rows = []

    for symbol in SYMBOLS:
        # Filter this symbol's indicator candles
        df_sym = extras[extras["symbol"] == symbol].sort_values("close_time")
        if df_sym.empty:
            continue

        # Align to future close
        df_sym = df_sym.assign(
            target_time=lambda d: d["close_time"] + pd.Timedelta(minutes=look_ahead)
        )
        future_tbl = (
            df_sym[["close_time", "close"]]
            .rename(columns={"close_time": "future_time", "close": "future_close"})
        )
        # Merge only exact lookahead matches
        df_sym = pd.merge_asof(
            df_sym,
            future_tbl,
            left_on="target_time",
            right_on="future_time",
            direction="forward",
        )
        # Keep only exact matches (no bias from far future)
        df_sym = df_sym[df_sym["future_time"] == df_sym["target_time"]]
        # Drop helper cols
        df_sym = df_sym.drop(columns=["future_time", "target_time"])
        if df_sym.empty:
            continue

        # Bucket label and set timestamp
        df_sym = df_sym.assign(
            pct_change=lambda d: (d["future_close"] - d["close"]) / d["close"],
            label=lambda d: d["pct_change"].apply(bucket_label),
            timestamp=lambda d: d["close_time"],
        )

        # Extract features
        for _, row in df_sym.iterrows():
            big_cnt, spread, diff_btc, diff_eth, diff_rndr = compute_feature_windows(
                extras, trades, orderbook, row, look_ahead
            )
            all_rows.append([
                row.timestamp,
                row.rsi,
                row.macd_hist,
                row.close,
                row.volume,
                big_cnt,
                spread,
                diff_btc,
                diff_eth,
                diff_rndr,
                row.label,
            ])

    columns = [
        "timestamp",
        "rsi",
        "macd_hist",
        "vib_close",
        "volume",
        "big_trades_count",
        "orderbook_spread",
        "diff_BTC",
        "diff_ETH",
        "diff_RNDR",
        "label",
    ]
    return pd.DataFrame(all_rows, columns=columns)


def main():
    extras = load_extras()
    trades = load_trades()
    orderbook = load_orderbook()

    if extras.empty:
        print("No extras data found. Exiting.")
        return

    df_train = build_dataset(extras, trades, orderbook, LOOK_AHEAD)
    if df_train.empty:
        print("No training rows generated. Exiting.")
        return

    with sqlite3.connect(TRAINING_DB) as conn:
        df_train.to_sql("merged_training_data", conn, if_exists="replace", index=False)

    print(f"Written {len(df_train)} rows to merged_training_data in {TRAINING_DB}")


if __name__ == "__main__":
    main()
