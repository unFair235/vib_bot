# config.py

import os
import json

# ────────────────────────────────────────────────────────────────────────────────
# Directories
# ────────────────────────────────────────────────────────────────────────────────
PACKAGE_DIR     = os.path.dirname(__file__)
# project root is one level above the package directory
BASE_DIR        = os.getenv(
    "VIB_BOT_BASE_DIR",
    os.path.abspath(os.path.join(PACKAGE_DIR, os.pardir))
)

# code folders inside the package
DATA_DIR        = os.path.join(PACKAGE_DIR, "data")
PROCESSING_DIR  = os.path.join(PACKAGE_DIR, "processing")
MODELS_DIR      = os.path.join(PACKAGE_DIR, "models")
REALTIME_DIR    = os.path.join(PACKAGE_DIR, "realtime")
BACKTEST_DIR    = os.path.join(PACKAGE_DIR, "backtest")
MONITORING_DIR  = os.path.join(PACKAGE_DIR, "monitoring")
UTILS_DIR       = os.path.join(PACKAGE_DIR, "utils")

# scripts and tests folders at project root
SCRIPTS_DIR     = os.path.join(BASE_DIR, "scripts")
TESTS_DIR       = os.path.join(BASE_DIR, "tests")

# symbols file at project root
SYMBOLS_FILE    = os.path.join(BASE_DIR, "symbols.json")

# ────────────────────────────────────────────────────────────────────────────────
# Database Files (at project root)
# ────────────────────────────────────────────────────────────────────────────────
EXTRAS_DB_FILE    = os.path.join(BASE_DIR, "vib_extra_data.db")
TRADES_DB_FILE    = os.path.join(BASE_DIR, "trades.db")
ORDERBOOK_DB_FILE = os.path.join(BASE_DIR, "orderbook.db")
MASTER_DB_FILE    = os.path.join(BASE_DIR, "vib_master.db")
TRAINING_DB_FILE  = os.path.join(BASE_DIR, "training_data.db")
METRICS_DB_FILE   = os.path.join(BASE_DIR, "metrics.db")

# ────────────────────────────────────────────────────────────────────────────────
# ML Artifacts
# ────────────────────────────────────────────────────────────────────────────────
MODEL_PATH_LINEAR   = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH_LINEAR  = os.path.join(MODELS_DIR, "scaler_linear.pkl")
MODEL_PATH_NN       = os.path.join(MODELS_DIR, "model_nn.keras")
SCALER_PATH_NN      = os.path.join(MODELS_DIR, "scaler_nn.pkl")
ACTIVE_MODEL_FILE   = os.path.join(MODELS_DIR, "active_model.txt")

# ────────────────────────────────────────────────────────────────────────────────
# Telegram / Notification
# ────────────────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN      = os.getenv("VIB_BOT_TELEGRAM_TOKEN", "")
CHAT_ID             = int(os.getenv("VIB_BOT_CHAT_ID", "123456789"))

# ────────────────────────────────────────────────────────────────────────────────
# Parameters: thresholds, windows & risk sizing
# ────────────────────────────────────────────────────────────────────────────────
BIG_TRADE_THRESHOLD      = int(os.getenv("VIB_BOT_BIG_TRADE_THRESHOLD", "100000"))
DATA_FRESHNESS_THRESHOLD = int(os.getenv("VIB_BOT_DATA_FRESHNESS_THRESHOLD", "120"))
FEEDBACK_WINDOW          = int(os.getenv("VIB_BOT_FEEDBACK_WINDOW", "3600"))
LOOK_AHEAD               = int(os.getenv("VIB_BOT_LOOK_AHEAD", "5"))
INFERENCE_INTERVAL       = int(os.getenv("VIB_BOT_INFERENCE_INTERVAL", "30"))

# Backtest realism
BACKTEST_SLIPPAGE_PCT    = float(os.getenv("VIB_BOT_BACKTEST_SLIPPAGE_PCT", "0.0005"))
BACKTEST_FEES_PCT        = float(os.getenv("VIB_BOT_BACKTEST_FEES_PCT", "0.001"))

# Execution‐Engine / Risk & Sizing
RISK_PER_TRADE           = float(os.getenv("VIB_BOT_RISK_PER_TRADE", "0.005"))
STOP_LOSS_PCT            = float(os.getenv("VIB_BOT_STOP_LOSS_PCT", "0.012"))
TAKE_PROFIT_PCT          = float(os.getenv("VIB_BOT_TAKE_PROFIT_PCT", "0.018"))
SLIPPAGE_TOLERANCE       = float(os.getenv("VIB_BOT_SLIPPAGE_TOLERANCE", "0.001"))
MAX_PORTFOLIO_DRAWDOWN   = float(os.getenv("VIB_BOT_MAX_DRAWDOWN", "0.20"))
COOLDOWN_HOURS           = int(os.getenv("VIB_BOT_COOLDOWN_HOURS", "2"))
MAX_DAILY_DRAWDOWN       = float(os.getenv("VIB_BOT_MAX_DAILY_DRAWDOWN", "0.05"))

# ────────────────────────────────────────────────────────────────────────────────
# Trading universe
# ────────────────────────────────────────────────────────────────────────────────
_env_syms = os.getenv("VIB_BOT_SYMBOLS")
if _env_syms:
    SYMBOLS = _env_syms.split(",")
else:
    try:
        with open(SYMBOLS_FILE) as f:
            SYMBOLS = json.load(f)
    except Exception:
        SYMBOLS = ["VIBUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT"]

# Re‐use SYMBOLS for correlation diffs
CORRELATION_SYMBOLS = SYMBOLS

# ────────────────────────────────────────────────────────────────────────────────
# Exchange / API credentials
# ────────────────────────────────────────────────────────────────────────────────
EXCHANGE_API_KEY      = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET   = os.getenv("EXCHANGE_API_SECRET", "")

# ────────────────────────────────────────────────────────────────────────────────
# Monitoring
# ────────────────────────────────────────────────────────────────────────────────
MONITORING_INTERVAL   = int(os.getenv("VIB_BOT_MONITORING_INTERVAL", "60"))
