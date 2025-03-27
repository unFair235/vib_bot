#!/usr/bin/env python3
import subprocess
import time
import logging
import os
from logging.handlers import RotatingFileHandler

# Define the base directory for your project
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

# Setup logging for the master process with log rotation
MASTER_LOG_FILE = os.path.join(BASE_DIR, "master.log")
logger = logging.getLogger("master")
logger.setLevel(logging.INFO)
rot_handler = RotatingFileHandler(MASTER_LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
rot_handler.setFormatter(formatter)
logger.addHandler(rot_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info("[MASTER] Starting master.py – Debug log enabled")
print("[MASTER] Starting master.py – Debug log enabled")

# Dictionary of script names and their absolute paths (including vib_extras)
SCRIPTS = {
    "vib_extras": os.path.join(BASE_DIR, "vib_extras.py"),
    "vib_alert": os.path.join(BASE_DIR, "vib_alert.py"),
    "vib_master": os.path.join(BASE_DIR, "vib_master.py"),
    "train_model_online_enhanced": os.path.join(BASE_DIR, "train_model_online_enhanced.py"),
    # "multi_socket": os.path.join(BASE_DIR, "multi_socket.py"),
}

# Corresponding log file paths for each script
LOG_FILES = {
    "vib_extras": os.path.join(BASE_DIR, "vib_extras.log"),
    "vib_alert": os.path.join(BASE_DIR, "vib_alert.log"),
    "vib_master": os.path.join(BASE_DIR, "vib_master.log"),
    "train_model_online_enhanced": os.path.join(BASE_DIR, "train_model_online_enhanced.log"),
    # "multi_socket": os.path.join(BASE_DIR, "multi_socket.log"),
}

# Dictionary to keep track of subprocesses and their log file objects
processes = {}

def start_script(name, script_path, log_path):
    """Starts a script as a subprocess with unbuffered output and logs its output."""
    logger.info(f"[MASTER] Starting {name}: {script_path}")
    log_file = open(log_path, "a", buffering=1)
    proc = subprocess.Popen(
        ["/usr/bin/python3", "-u", script_path],
        stdout=log_file,
        stderr=log_file,
        cwd=BASE_DIR
    )
    processes[name] = (proc, log_file)

def monitor_processes():
    """Checks every 30 seconds if any launched process has terminated, and restarts it if needed."""
    while True:
        for name, (proc, log_file) in list(processes.items()):
            ret = proc.poll()
            if ret is not None:
                logger.warning(f"[MASTER] {name} terminated (exit code {ret}). Restarting...")
                log_file.close()
                start_script(name, SCRIPTS[name], LOG_FILES[name])
        time.sleep(30)

if __name__ == "__main__":
    for name, script_path in SCRIPTS.items():
        start_script(name, script_path, LOG_FILES[name])
    monitor_processes()