#!/usr/bin/env bash
set -euo pipefail
PIDFILE="${TMPDIR:-/tmp}/archimedes-cloudflared.pid"
if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" && echo "Stopped cloudflared (PID $pid)"
  fi
  rm -f "$PIDFILE"
else
  echo "No pidfile at $PIDFILE"
fi
rm -f "${TMPDIR:-/tmp}/archimedes-tunnel.public_base" 2>/dev/null || true
