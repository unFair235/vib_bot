#!/usr/bin/env bash
set -euo pipefail

# Go to project root (parent of scripts/)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

echo "🛑 Stopping master and daemons…"
pkill -f scripts/master.py                         2>/dev/null || true
pkill -f data/vib_extras.py                        2>/dev/null || true
pkill -f realtime/multi_socket.py                  2>/dev/null || true
pkill -f realtime/vib_master.py                    2>/dev/null || true
pkill -f models/train_model_online_enhanced.py     2>/dev/null || true
pkill -f data/vib_orderbook.py                     2>/dev/null || true
sleep 2

BACKUP_SUFFIX=".bak"
ARTIFACTS=(
  "models/model.pkl"
  "models/scaler_linear.pkl"
  "models/model_nn.keras"
  "models/scaler_nn.pkl"
  "models/active_model.txt"
)

echo "↩️  Rolling back model artifacts from backups…"
for file in "${ARTIFACTS[@]}"; do
  if [ -f "$file$BACKUP_SUFFIX" ]; then
    echo "Restoring $file from $file$BACKUP_SUFFIX"
    mv -f "$file$BACKUP_SUFFIX" "$file"
  else
    echo "No backup found for $file, skipping"
  fi
done

echo "✅ Rollback complete."

echo "🚀 Restarting master…"
python3 scripts/master.py &> master.startup.log &
echo "Master PID: $!"

exit 0