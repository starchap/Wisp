#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "============================================================================"
echo "  Wisp — 100% Local AI Meeting Transcriber & Translator"
echo "============================================================================"

echo "[1/2] Starting Local Python AI Engine on http://127.0.0.1:8765 ..."
python3 backend/main.py &
BACKEND_PID=$!

trap "kill $BACKEND_PID 2>/dev/null || true" EXIT
sleep 2

echo "[2/2] Launching Wisp Desktop Application ..."
cd frontend
export SKIP_BACKEND_SPAWN=1
export USE_DIST=1
if [ -x "./node_modules/.bin/electron" ]; then
  ./node_modules/.bin/electron .
else
  npm run dev:vite
fi
