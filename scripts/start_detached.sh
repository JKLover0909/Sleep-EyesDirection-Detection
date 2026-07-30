#!/usr/bin/env bash
# Start FQC demo in a detached tmux session (survives SSH/Cursor disconnect).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="fqc-sleep-eyes"
LOG="$ROOT/data/server.log"
PIDFILE="$ROOT/data/server.pid"
PY="$ROOT/.venv/bin/python"
APP="$ROOT/backend/main.py"

mkdir -p "$ROOT/data"
touch "$LOG"

# Stop anything already bound to this app / port 8010 from previous runs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION" || true
fi
if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    kill "$old" 2>/dev/null || true
    sleep 1
    kill -9 "$old" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi
# Best-effort: free port 8010 if leftover python holds it
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8010/tcp 2>/dev/null || true
fi

# Restart loop so crash/reboot-of-process keeps the demo live until machine reboot
tmux new-session -d -s "$SESSION" -c "$ROOT/backend" \
  "echo \$\$ > '$PIDFILE'; \
   while true; do \
     echo \"[\$(date -Is)] starting FQC demo\" >> '$LOG'; \
     '$PY' '$APP' >> '$LOG' 2>&1; \
     code=\$?; \
     echo \"[\$(date -Is)] exited code=\$code — restart in 3s\" >> '$LOG'; \
     sleep 3; \
   done"

sleep 2
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Failed to start tmux session $SESSION" >&2
  exit 1
fi

echo "Started session: $SESSION"
echo "Log:            $LOG"
echo "Dashboard:      http://0.0.0.0:8010  (or http://<server-ip>:8010)"
echo
echo "Useful commands:"
echo "  tmux attach -t $SESSION     # xem log live (Ctrl+B D để detach)"
echo "  $ROOT/scripts/status.sh"
echo "  $ROOT/scripts/stop.sh"
