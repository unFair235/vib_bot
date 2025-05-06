#!/usr/bin/env bash
set -euo pipefail

# jump to the directory where THIS script lives
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 0. Tear down any existing master & child scripts
echo "🔄 Shutting down any running master or child processes…"
pkill -f master.py                2>/dev/null || true
pkill -f vib_extras.py            2>/dev/null || true
pkill -f multi_socket.py          2>/dev/null || true
pkill -f vib_master.py            2>/dev/null || true
pkill -f train_model_online_enhanced.py 2>/dev/null || true
pkill -f vib_orderbook.py         2>/dev/null || true
sleep 2

# 1. (We’re already in the project root now)

# 2. Launch master in the background
echo "🚀 Starting master.py…"
bash -c "./master.py &> master.startup.log & echo \$! > master.pid"

# Give everything a few seconds to spin up
sleep 10

# 3. Check that each subprocess is running
echo && echo "=== Running Processes ==="
MASTER_PID=$(< master.pid)
ps -o pid,cmd -p "${MASTER_PID}" \
    -p "$(pgrep -f vib_extras.py)" \
    -p "$(pgrep -f multi_socket.py)" \
    -p "$(pgrep -f vib_master.py)" \
    -p "$(pgrep -f train_model_online_enhanced.py)" \
    -p "$(pgrep -f vib_orderbook.py)" \
|| true

# 4. Tail the last lines of each log to look for errors
echo && echo "=== Last 5 lines of each log ==="
for log in vib_extras.log multi_socket.log vib_master.log train_model_online_enhanced.log vib_orderbook.log; do
  echo "--- $log ---"
  tail -n 5 "$log" 2>/dev/null || echo "(no file)"
done

# 5. Sanity‐check your vib_master.db contents
echo && echo "=== DB Table Counts (vib_master.db) ==="
sqlite3 vib_master.db << 'EOF'
.headers on
.mode column
SELECT 'predictions',       COUNT(*) FROM predictions;
SELECT 'pending_feedback',  COUNT(*) FROM pending_feedback;
SELECT 'feedback',          COUNT(*) FROM feedback;
EOF

# 6. Shut down the master process
echo && echo "🛑 Stopping master (PID $MASTER_PID)…"
kill "${MASTER_PID}" 2>/dev/null || true

# 7. Exit cleanly so Automator/bash sees “success”
exit 0