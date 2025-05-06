#!/usr/bin/env python3
"""
scripts/master.py

1) symbols update (file)
2) generate merged_training_data (module)
3) offline trainers  (modules)
4) model switch      (module)
5) launch daemons    (mix of file + module)
"""

import subprocess, time, logging, os, signal, sys
from logging.handlers import RotatingFileHandler

from vib_bot.config import (
    BASE_DIR,
    MODEL_PATH_LINEAR,
    SCALER_PATH_LINEAR,
    MODEL_PATH_NN,
    SCALER_PATH_NN,
)

# ─── File scripts ────────────────────────────────────────────────────────────
UPDATE_SYMBOLS = os.path.join(BASE_DIR, "scripts", "update_symbols.py")

# ─── Module invocations ──────────────────────────────────────────────────────
GENERATE_MODULE    = "vib_bot.processing.generate_training_data"
LINEAR_MODULE      = "vib_bot.models.train_model_linear"
NN_MODULE          = "vib_bot.models.train_model_nn"
SWITCH_MODULE      = "vib_bot.models.evaluate_and_switch"

# ─── Logging ─────────────────────────────────────────────────────────────────
MASTER_LOG = os.path.join(BASE_DIR, "master.log")
logger = logging.getLogger("master")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

rot = RotatingFileHandler(MASTER_LOG, maxBytes=5*1024*1024, backupCount=3)
rot.setFormatter(fmt)
logger.addHandler(rot)

stream = logging.StreamHandler()
stream.setFormatter(fmt)
logger.addHandler(stream)


def run_script(path, desc):
    logger.info(f"🔨  {desc}…")
    rc = subprocess.call([sys.executable, path], cwd=BASE_DIR)
    if rc:
        logger.error(f"{desc} failed (exit {rc}); aborting.")
        sys.exit(1)
    logger.info(f"✅  {desc} completed.")


def run_module(module, desc):
    logger.info(f"🔨  {desc}…")
    rc = subprocess.call([sys.executable, "-m", module], cwd=BASE_DIR)
    if rc:
        logger.error(f"{desc} failed (exit {rc}); aborting.")
        sys.exit(1)
    logger.info(f"✅  {desc} completed.")


# ─── 0) Refresh your symbol list ──────────────────────────────────────────────
run_script(UPDATE_SYMBOLS, "Updating symbols.json")

# ─── 1) Build the merged_training_data via your ETL ──────────────────────────
run_module(GENERATE_MODULE, "Generating merged_training_data")

# ─── 2) If linear model missing, train one ────────────────────────────────────
if not os.path.exists(MODEL_PATH_LINEAR) or not os.path.exists(SCALER_PATH_LINEAR):
    run_module(LINEAR_MODULE, "Offline linear trainer")

# ─── 3) If NN model missing, train one ────────────────────────────────────────
if not os.path.exists(MODEL_PATH_NN) or not os.path.exists(SCALER_PATH_NN):
    run_module(NN_MODULE, "Offline NN trainer")

# ─── 4) Evaluate & switch active model ───────────────────────────────────────
logger.info("🔄  Evaluating & switching active model…")
rc = subprocess.call([sys.executable, "-m", SWITCH_MODULE], cwd=BASE_DIR)
if rc:
    logger.warning(f"evaluate_and_switch exited {rc}; keeping current model.")
else:
    logger.info("✅  Model switch step completed.")


# ─── 5) Daemons: mix of file scripts and modules ───────────────────────────────
SCRIPTS = {
    "update_symbols": UPDATE_SYMBOLS,
    "vib_extras":     ("module", "vib_bot.data.vib_extras"),
    "multi_socket":   ("module", "vib_bot.realtime.multi_socket"),
    "vib_master":     ("module", "vib_bot.realtime.vib_master"),
    "train_online":   ("module", "vib_bot.models.train_model_online_enhanced"),
}
LOGS = {name: os.path.join(BASE_DIR, f"{name}.log") for name in SCRIPTS}
processes = {}


def start_service(name, entry, logpath):
    logger.info(f"[MASTER] Starting {name}")
    f = open(logpath, "a", buffering=1)
    if isinstance(entry, tuple) and entry[0] == "module":
        cmd = [sys.executable, "-u", "-m", entry[1]]
    else:
        cmd = [sys.executable, "-u", entry]
    p = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=f, stderr=f)
    processes[name] = (p, f)


def monitor():
    while True:
        for name, (p, f) in list(processes.items()):
            if p.poll() is not None:
                logger.warning(f"[MASTER] {name} exited ({p.returncode}); restarting…")
                f.close()
                start_service(name, SCRIPTS[name], LOGS[name])
        time.sleep(30)


def shutdown(sig, frame):
    logger.info("[MASTER] Shutting down, terminating children…")
    for p, f in processes.values():
        p.terminate()
        f.close()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


if __name__ == "__main__":
    logger.info("[MASTER] Launching managed services…")
    for name, entry in SCRIPTS.items():
        start_service(name, entry, LOGS[name])
    monitor()