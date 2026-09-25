@echo off
title LayaQuant Studio
echo =======================================================
echo         Starting LayaQuant Studio (100%% Open-Source)
echo =======================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo Starting LayaQuant Decision Daemon on http://127.0.0.1:8000 ...
echo Loading model engine and initializing API endpoints...
start "" http://127.0.0.1:8000/plan
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
pause
