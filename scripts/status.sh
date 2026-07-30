#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="fqc-sleep-eyes"
PIDFILE="$ROOT/data/server.pid"
LOG="$ROOT/data/server.log"

echo "=== tmux ==="
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session: $SESSION (running)"
  tmux list-sessions | grep "$SESSION" || true
else
  echo "session: $SESSION (not found)"
fi

echo
echo "=== process / port ==="
if [[ -f "$PIDFILE" ]]; then
  echo "pidfile: $(cat "$PIDFILE")"
fi
pgrep -af "Sleep-EyesDirection-Detection/backend/main.py|backend/main.py" || echo "no python main.py matched"
ss -ltnp 2>/dev/null | grep 8010 || netstat -ltnp 2>/dev/null | grep 8010 || echo "port 8010 not listening"

echo
echo "=== api ==="
if curl -fsS --max-time 2 http://127.0.0.1:8010/api/stats >/tmp/fqc_stats.json 2>/dev/null; then
  python3 - <<'PY'
import json
d=json.load(open("/tmp/fqc_stats.json"))
print("system:", d.get("system"), "| fps:", d.get("fps"), "| face:", d.get("face_detected"), "| gaze:", d.get("gaze_direction"))
PY
else
  echo "API not reachable on :8010"
fi

echo
echo "=== last log lines ==="
tail -n 15 "$LOG" 2>/dev/null || echo "(no log yet)"
