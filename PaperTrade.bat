@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo ProAsh - checking open paper trades against latest close...
echo ============================================================
python track_picks.py

echo.
pause
