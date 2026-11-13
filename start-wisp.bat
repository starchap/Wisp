@echo off
setlocal
title Wisp

cd /d "%~dp0"

echo [1/2] Starting Wisp Local Python AI Engine on http://127.0.0.1:8765 ...
start "Wisp Local AI Engine" /min py -3.10 backend\main.py

timeout /t 2 /nobreak >nul

echo [2/2] Launching Wisp Desktop Application ...
set SKIP_BACKEND_SPAWN=1
set USE_DIST=1
start "" /d "%~dp0frontend" "%~dp0frontend\node_modules\electron\dist\electron.exe" .
