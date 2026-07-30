#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="fqc-sleep-eyes"
PIDFILE="$ROOT/data/server.pid"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "Stopped tmux session: $SESSION"
else
  echo "No tmux session: $SESSION"
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

if command -v fuser >/dev/null 2>&1; then
  fuser -k 8010/tcp 2>/dev/null || true
fi

echo "Done."
