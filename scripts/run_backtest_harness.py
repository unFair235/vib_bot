#!/usr/bin/env python3
import vib_bot.config as cfg
from vib_bot.realtime.vib_master      import load_extras, load_trades, load_orderbook, load_active_model
from vib_bot.backtest.backtest_harness import run_backtest_for_symbol

def main():
    extras, trades, orderbook = load_extras(), load_trades(), load_orderbook()
    (model, scaler), model_id = load_active_model()
    rets, metrics = run_backtest_for_symbol(
        extras, trades, orderbook,
        cfg.SYMBOLS[0], model, scaler
    )
    print(f"{model_id} → {metrics}")

if __name__ == "__main__":
    main()
