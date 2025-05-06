#!/usr/bin/env bash
set -euo pipefail

# 0. cd to project root (parent of this script)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

echo "🔄 Shutting down any running master or child processes…"
pkill -f scripts/master.py                        2>/dev/null || true
pkill -f data/vib_extras.py                       2>/dev/null || true
pkill -f realtime/multi_socket.py                 2>/dev/null || true
pkill -f realtime/vib_master.py                   2>/dev/null || true
pkill -f models/train_model_online_enhanced.py    2>/dev/null || true
pkill -f data/vib_orderbook.py                    2>/dev/null || true
sleep 2

echo "🚀 Starting master.py…"
# run master and capture its PID
python3 scripts/master.py &> master.startup.log &
echo $! > master.pid

# give it a moment to spin up
sleep 10

echo && echo "=== Processes by script ==="
for script in \
  scripts/master.py \
  data/vib_extras.py \
  realtime/multi_socket.py \
  realtime/vib_master.py \
  models/train_model_online_enhanced.py \
  data/vib_orderbook.py; do
  printf "%-50s" "$script:"
  if pgrep -f "python.*[ /]$script" >/dev/null; then
    echo " RUNNING"
  else
    echo " NOT RUNNING"
  fi
done

echo && echo "✅ Setup verification complete."