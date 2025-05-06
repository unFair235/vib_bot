# config.py

import os
import json

# ────────────────────────────────────────────────────────────────────────────────
# Base & Sub‐folder Paths
# ────────────────────────────────────────────────────────────────────────────────
BASE_DIR         = os.getenv("VIB_BOT_BASE_DIR", os.path.expanduser("~/Documents/vib_bot"))
DATA_DIR         = os.path.join(BASE_DIR, "data")
PROCESSING_DIR   = os.path.join(BASE_DIR, "processing")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
REALTIME_DIR     = os.path.join(BASE_DIR, "realtime")
BACKTEST_DIR     = os.path.join(BASE_DIR, "backtest")
MONITORING_DIR   = os.path.join(BASE_DIR, "monitoring")
SCRIPTS_DIR      = os.path.join(BASE_DIR, "scripts")
TESTS_DIR        = os.path.join(BASE_DIR, "tests")

# ────────────────────────────────────────────────────────────────────────────────
# Database Files (still in BASE_DIR)
# ────────────────────────────────────────────────────────────────────────────────
EXTRAS_DB_FILE        = os.path.join(BASE_DIR, "vib_extra_data.db")
TRADES_DB_FILE        = os.path.join(BASE_DIR, "trades.db")
ORDERBOOK_DB_FILE     = os.path.join(BASE_DIR, "orderbook.db")
MASTER_DB_FILE        = os.path.join(BASE_DIR, "vib_master.db")
TRAINING_DB_FILE      = os.path.join(BASE_DIR, "training_data.db")
METRICS_DB_FILE       = os.path.join(BASE_DIR, "metrics.db")

# ────────────────────────────────────────────────────────────────────────────────
# ML Artifacts
# ────────────────────────────────────────────────────────────────────────────────
MODEL_PATH_LINEAR     = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH_LINEAR    = os.path.join(MODELS_DIR, "scaler_linear.pkl")
MODEL_PATH_NN         = os.path.join(MODELS_DIR, "model_nn.keras")
SCALER_PATH_NN        = os.path.join(MODELS_DIR, "scaler_nn.pkl")
ACTIVE_MODEL_FILE     = os.path.join(MODELS_DIR, "active_model.txt")

# ────────────────────────────────────────────────────────────────────────────────
# Script Locations (for master.py orchestration)
# ────────────────────────────────────────────────────────────────────────────────
GENERATE_SCRIPT       = os.path.join(PROCESSING_DIR, "generate_training_data.py")
TRAINER_LINEAR        = os.path.join(MODELS_DIR, "train_model_linear.py")
TRAINER_NN            = os.path.join(MODELS_DIR, "train_model_nn.py")
EVAL_SWITCH_SCRIPT    = os.path.join(MODELS_DIR, "evaluate_and_switch.py")
ONLINE_UPDATER_SCRIPT = os.path.join(MODELS_DIR, "train_model_online_enhanced.py")

# ────────────────────────────────────────────────────────────────────────────────
# Live‐feed & realtime files
# ────────────────────────────────────────────────────────────────────────────────
VIB_EXTRAS_SCRIPT     = os.path.join(DATA_DIR, "vib_extras.py")
VIB_ALERT_SCRIPT      = os.path.join(DATA_DIR, "vib_alert.py")
VIB_ORDERBOOK_SCRIPT  = os.path.join(DATA_DIR, "vib_orderbook.py")
MASTER_RT_SCRIPT      = os.path.join(REALTIME_DIR, "vib_master.py")
EXECUTOR_SCRIPT       = os.path.join(REALTIME_DIR, "execute_trades.py")
RISK_MANAGER_SCRIPT   = os.path.join(REALTIME_DIR, "risk_manager.py")
TRACKER_SCRIPT        = os.path.join(REALTIME_DIR, "trade_tracker.py")
MULTI_SOCKET_SCRIPT   = os.path.join(REALTIME_DIR, "multi_socket.py")

# ────────────────────────────────────────────────────────────────────────────────
# Orchestration & DevOps
# ────────────────────────────────────────────────────────────────────────────────
MASTER_SCRIPT         = os.path.join(SCRIPTS_DIR, "master.py")
VERIFY_SCRIPT         = os.path.join(SCRIPTS_DIR, "verify_setup.sh")
ROLLBACK_SCRIPT       = os.path.join(SCRIPTS_DIR, "rollback_models.sh")

# ────────────────────────────────────────────────────────────────────────────────
# Monitoring & Backtest
# ────────────────────────────────────────────────────────────────────────────────
BACKTEST_MODEL        = os.path.join(BACKTEST_DIR, "backtest_model.py")
BACKTEST_UTILS        = os.path.join(BACKTEST_DIR, "backtest_utils.py")
METRICS_COLLECTOR     = os.path.join(MONITORING_DIR, "metrics_collector.py")
DASHBOARD_APP         = os.path.join(MONITORING_DIR, "dashboard.py")

# ────────────────────────────────────────────────────────────────────────────────
# Testing
# ────────────────────────────────────────────────────────────────────────────────
TEST_GENERATE         = os.path.join(TESTS_DIR, "test_generate_training_data.py")
TEST_MERGE            = os.path.join(TESTS_DIR, "test_merge_data.py")
TEST_MODELS           = os.path.join(TESTS_DIR, "test_models.py")
TEST_REALTIME         = os.path.join(TESTS_DIR, "test_realtime.py")

# ────────────────────────────────────────────────────────────────────────────────
# Telegram / Notification
# ────────────────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN        = os.getenv("VIB_BOT_TELEGRAM_TOKEN", "")
CHAT_ID               = int(os.getenv("VIB_BOT_CHAT_ID", "123456789"))

# ────────────────────────────────────────────────────────────────────────────────
# Parameters: thresholds, windows & risk sizing
# ────────────────────────────────────────────────────────────────────────────────
BIG_TRADE_THRESHOLD       = int(os.getenv("VIB_BOT_BIG_TRADE_THRESHOLD", "100000"))
DATA_FRESHNESS_THRESHOLD  = int(os.getenv("VIB_BOT_DATA_FRESHNESS_THRESHOLD", "120"))
FEEDBACK_WINDOW           = int(os.getenv("VIB_BOT_FEEDBACK_WINDOW", "3600"))
LOOK_AHEAD                = int(os.getenv("VIB_BOT_LOOK_AHEAD", "5"))
INFERENCE_INTERVAL        = int(os.getenv("VIB_BOT_INFERENCE_INTERVAL", "30"))

# Backtest realism
BACKTEST_SLIPPAGE_PCT     = float(os.getenv("VIB_BOT_BACKTEST_SLIPPAGE_PCT", "0.0005"))
BACKTEST_FEES_PCT         = float(os.getenv("VIB_BOT_BACKTEST_FEES_PCT",     "0.001"))

# Execution‐Engine / Risk & Sizing
RISK_PER_TRADE            = float(os.getenv("VIB_BOT_RISK_PER_TRADE",     "0.005"))
STOP_LOSS_PCT             = float(os.getenv("VIB_BOT_STOP_LOSS_PCT",      "0.012"))
TAKE_PROFIT_PCT           = float(os.getenv("VIB_BOT_TAKE_PROFIT_PCT",    "0.018"))
# live trading slippage guard
SLIPPAGE_TOLERANCE        = float(os.getenv("VIB_BOT_SLIPPAGE_TOLERANCE", "0.001"))
# portfolio drawdown breaker
MAX_PORTFOLIO_DRAWDOWN    = float(os.getenv("VIB_BOT_MAX_DRAWDOWN",      "0.20"))

COOLDOWN_HOURS            = int(os.getenv("VIB_BOT_COOLDOWN_HOURS",      "2"))
MAX_DAILY_DRAWDOWN        = float(os.getenv("VIB_BOT_MAX_DAILY_DRAWDOWN","0.05"))

# ────────────────────────────────────────────────────────────────────────────────
# Trading universe
# ────────────────────────────────────────────────────────────────────────────────
_env_syms = os.getenv("VIB_BOT_SYMBOLS")
if _env_syms:
    SYMBOLS = _env_syms.split(",")
else:
    symbols_file = os.path.join(BASE_DIR, "symbols.json")
    try:
        with open(symbols_file) as f:
            SYMBOLS = json.load(f)
    except Exception:
        # fallback to a minimal list if symbols.json missing
        SYMBOLS = ["VIBUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT"]

# Correlation symbols for feature‐window diffs
CORRELATION_SYMBOLS = SYMBOLS

# ────────────────────────────────────────────────────────────────────────────────
# Exchange / API credentials
# ────────────────────────────────────────────────────────────────────────────────
EXCHANGE_API_KEY          = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET       = os.getenv("EXCHANGE_API_SECRET", "")

# ────────────────────────────────────────────────────────────────────────────────
# Monitoring
# ────────────────────────────────────────────────────────────────────────────────
MONITORING_INTERVAL       = int(os.getenv("VIB_BOT_MONITORING_INTERVAL", "60"))