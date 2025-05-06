#!/usr/bin/env python3
import subprocess
import time
import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler

from vib_bot.config import (
    BASE_DIR,
    MODEL_PATH_LINEAR,
    SCALER_PATH_LINEAR,
    MODEL_PATH_NN,
    SCALER_PATH_NN,
)

# ─── Paths for your orchestration scripts ────────────────────────────────────
UPDATE_SYMBOLS_SCRIPT = os.path.join(BASE_DIR, "scripts", "update_symbols.py")
GENERATE_SCRIPT       = os.path.join(BASE_DIR, "processing", "generate_training_data.py")
TRAINER_LINEAR        = os.path.join(BASE_DIR, "models", "train_model_linear.py")
TRAINER_NN            = os.path.join(BASE_DIR, "models", "train_model_nn.py")
EVAL_SWITCH_SCRIPT    = os.path.join(BASE_DIR, "models", "evaluate_and_switch.py")

# ─── Logger Setup ────────────────────────────────────────────────────────────
MASTER_LOG_FILE = os.path.join(BASE_DIR, "master.log")
logger = logging.getLogger("master")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")

rot_handler = RotatingFileHandler(MASTER_LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
rot_handler.setFormatter(fmt)
logger.addHandler(rot_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(fmt)
logger.addHandler(stream_handler)

def run_script(path, description):
    logger.info(f"🔨  {description}…")
    rc = subprocess.call([sys.executable, path], cwd=BASE_DIR)
    if rc != 0:
        logger.error(f"{description} failed (exit code {rc}); aborting master startup.")
        sys.exit(1)
    logger.info(f"✅  {description} completed successfully.")

# ─── Step 0: Update symbol list ───────────────────────────────────────────────
run_script(UPDATE_SYMBOLS_SCRIPT, "Updating symbols.json")

# ─── Step 1: Generate merged training data ───────────────────────────────────
run_script(GENERATE_SCRIPT, "Generating merged_training_data")

# ─── Step 2: Bootstrap linear model if missing ───────────────────────────────
if not os.path.exists(MODEL_PATH_LINEAR) or not os.path.exists(SCALER_PATH_LINEAR):
    run_script(TRAINER_LINEAR, "Offline linear trainer")

# ─── Step 3: Bootstrap neural‑net model if missing ────────────────────────────
if not os.path.exists(MODEL_PATH_NN) or not os.path.exists(SCALER_PATH_NN):
    run_script(TRAINER_NN, "Offline NN trainer")

# ─── Step 4: Auto‑switch active model ─────────────────────────────────────────
logger.info("🔄  Evaluating & switching active model if needed…")
rc = subprocess.call([sys.executable, EVAL_SWITCH_SCRIPT], cwd=BASE_DIR)
if rc != 0:
    logger.warning(f"`evaluate_and_switch.py` exited with code {rc}; keeping existing active_model.txt")
else:
    logger.info("✅  `evaluate_and_switch.py` ran successfully.")

# ─── Daemons under master ────────────────────────────────────────────────────
SCRIPTS = {
    "update_symbols":     UPDATE_SYMBOLS_SCRIPT,
    "vib_extras":         os.path.join(BASE_DIR, "data", "vib_extras.py"),
    "multi_socket":       os.path.join(BASE_DIR, "realtime", "multi_socket.py"),
    "vib_master":         os.path.join(BASE_DIR, "realtime", "vib_master.py"),
    "train_model_online": os.path.join(BASE_DIR, "models", "train_model_online_enhanced.py"),
}
LOG_FILES = {name: os.path.join(BASE_DIR, f"{name}.log") for name in SCRIPTS}
processes = {}

def start_script(name, path, log_path):
    logger.info(f"[MASTER] Starting {name}")
    f = open(log_path, "a", buffering=1)
    p = subprocess.Popen([sys.executable, "-u", path], stdout=f, stderr=f, cwd=BASE_DIR)
    processes[name] = (p, f)

def monitor():
    while True:
        for name, (p, f) in list(processes.items()):
            if p.poll() is not None:
                logger.warning(f"[MASTER] {name} exited ({p.returncode}); restarting…")
                f.close()
                start_script(name, SCRIPTS[name], LOG_FILES[name])
        time.sleep(30)

def shutdown(signum, frame):
    logger.info("[MASTER] Shutting down, terminating children…")
    for p, f in processes.values():
        p.terminate()
        f.close()
    sys.exit(0)

# Trap SIGINT/SIGTERM so Ctrl‑C cleans up children
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == "__main__":
    logger.info("[MASTER] Launching all managed scripts")
    for name, path in SCRIPTS.items():
        start_script(name, path, LOG_FILES[name])
    monitor()