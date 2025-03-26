#!/usr/bin/env python3
import subprocess
import time
import logging
import os

# Define the base directory for your project.
BASE_DIR = "/Users/igorbulgakov/Documents/vib_bot"

# Absolute paths for all your scripts (located in the project directory)
SCRIPTS = {
    "vib_extras": f"{BASE_DIR}/vib_extras.py",
    "multi_socket": f"{BASE_DIR}/multi_socket.py",
    "train_model_online_enhanced": f"{BASE_DIR}/train_model_online_enhanced.py",
    "vib_master": f"{BASE_DIR}/vib_master.py",
}

# Corresponding absolute log file paths (stored in the project directory)
LOG_FILES = {
    "vib_extras": f"{BASE_DIR}/vib_extras.log",
    "multi_socket": f"{BASE_DIR}/multi_socket.log",
    "train_model_online_enhanced": f"{BASE_DIR}/train_model_online_enhanced.log",
    "vib_master": f"{BASE_DIR}/vib_master.log",
}

# Master log file path
MASTER_LOG_FILE = f"{BASE_DIR}/master.log"

# Setup logging for the master script.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(MASTER_LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("master")

# Dictionary to manage subprocesses and their log file objects.
processes = {}

def start_script(name, script_path, log_path):
    """Start a script as a subprocess with immediate (unbuffered) output to a log file."""
    logger.info(f"[MASTER] Starting {name}: {script_path}")
    # Open the log file in line-buffered mode.
    log_file = open(log_path, "a", buffering=1)
    proc = subprocess.Popen(
        ["/usr/bin/python3", "-u", script_path],  # Use unbuffered mode (-u)
        stdout=log_file,
        stderr=log_file,
        cwd=BASE_DIR  # Set working directory to the project folder
    )
    processes[name] = (proc, log_file)

def monitor_processes():
    """Periodically check all subprocesses; restart any that have terminated."""
    while True:
        for name, (proc, log_file) in list(processes.items()):
            ret = proc.poll()
            if ret is not None:
                logger.warning(f"[MASTER] {name} terminated (exit {ret}). Restarting...")
                log_file.close()
                start_script(name, SCRIPTS[name], LOG_FILES[name])
        time.sleep(30)

if __name__ == "__main__":
    logger.info("[MASTER] master.py script STARTED")
    
    # Launch each of the scripts initially.
    for name, script_path in SCRIPTS.items():
        start_script(name, script_path, LOG_FILES[name])
    
    # Begin monitoring and restarting terminated processes.
    monitor_processes()