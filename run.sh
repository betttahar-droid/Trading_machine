#!/usr/bin/env bash
set -e
echo "======================================================="
echo "       Starting LayaQuant Studio (100% Open-Source)    "
echo "======================================================="
python3 -m pip install -r backend/requirements.txt --quiet
echo "Starting LayaQuant Daemon on http://127.0.0.1:8000 ..."
if which xdg-open > /dev/null; then
  xdg-open http://127.0.0.1:8000/plan &
elif which open > /dev/null; then
  open http://127.0.0.1:8000/plan &
fi
python3 -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
