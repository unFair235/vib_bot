#!/usr/bin/env python3
import os
import json
import sys
import requests
from typing import List

from vib_bot.config import BASE_DIR, SYMBOLS as DEFAULT_SYMBOLS

CMC_API_KEY = os.getenv("CMC_API_KEY")
CMC_API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
OUT_PATH    = os.path.join(BASE_DIR, "symbols.json")

def fetch_bottom50_of_top100(api_key: str) -> List[str]:
    if not api_key:
        raise RuntimeError("CMC_API_KEY not set; cannot fetch updated symbols.")
    params = {"start": 1, "limit": 100, "convert": "USD"}
    headers = {"X-CMC_PRO_API_KEY": api_key}
    resp = requests.get(CMC_API_URL, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    # Append "USDT" to each symbol and take the bottom 50
    symbols = [item["symbol"].upper() + "USDT" for item in data if "symbol" in item]
    return symbols[-50:]

def main():
    try:
        syms = fetch_bottom50_of_top100(CMC_API_KEY)
        print(f"✅ Fetched {len(syms)} symbols from CMC.")
    except Exception as e:
        print(f"⚠️  Failed to fetch from CMC: {e}", file=sys.stderr)
        syms = DEFAULT_SYMBOLS
        print(f"ℹ️  Falling back to default SYMBOLS from config: {syms}", file=sys.stderr)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Write out to symbols.json
    with open(OUT_PATH, "w") as f:
        json.dump(syms, f, indent=2)

    print(f"📦 Wrote {len(syms)} symbols to {OUT_PATH}")

if __name__ == "__main__":
    main()