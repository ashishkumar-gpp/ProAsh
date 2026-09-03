@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo ProAsh - refreshing NSE+BSE bhavcopy data...
echo ============================================================
python engine\download_data.py
if errorlevel 1 (
    echo.
    echo [warning] Data refresh failed or was skipped - continuing with existing local data.
)

echo.
echo ============================================================
echo ProAsh - running ensemble pipeline...
echo ============================================================
python proash_pipeline.py

echo.
pause
