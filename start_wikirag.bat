@echo off
title WikiRAG - Tensura Knowledge Base
cd /d "%~dp0"

echo ========================================================
echo         WikiRAG - Tensura Knowledge System
echo ========================================================
echo.
echo Starting WikiRAG Server at http://localhost:8000 ...
echo.

REM Open browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"

REM Run FastAPI server
python -m wikirag serve --host 127.0.0.1 --port 8000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start server.
    pause
)
