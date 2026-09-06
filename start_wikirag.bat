@echo off
title WikiRAG - Tensura Knowledge Base
cd /d "%~dp0"

echo ========================================================
echo         WikiRAG - Tensura Knowledge System
echo ========================================================
echo.
echo Starting WikiRAG Server at http://localhost:8000 ...
echo.

REM Reuse an already-running server instead of binding port 8000 twice.
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul
if %ERRORLEVEL% EQU 0 (
    echo WikiRAG is already running at http://localhost:8000
    echo The existing server owns port 8000, so no second server was started.
    start "" http://localhost:8000
    echo.
    echo This window will stay open so you can read the status above.
    pause
    exit /b 0
)

REM Open browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"

REM Run FastAPI server
python -m wikirag serve --host 127.0.0.1 --port 8000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start server.
    pause
)
