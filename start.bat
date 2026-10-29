@echo off
title TruthLens - AI Fake News Detector
echo ========================================================
echo       Starting TruthLens AI Fake News Detector
echo ========================================================
echo.

set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%backend" (
    set "BACKEND_DIR=%SCRIPT_DIR%backend"
    set "FRONTEND_DIR=%SCRIPT_DIR%frontend"
) else (
    set "BACKEND_DIR=%SCRIPT_DIR%New folder\backend"
    set "FRONTEND_DIR=%SCRIPT_DIR%New folder\frontend"
)

echo [1/3] Starting Python ML Service (FastAPI on http://127.0.0.1:8001)...
start "TruthLens ML Service" /D "%BACKEND_DIR%" cmd /k "python -m uvicorn ml_service.main:app --host 127.0.0.1 --port 8001"

timeout /t 3 /nobreak >nul

echo [2/3] Starting Vite React Frontend (http://localhost:5173)...
start "TruthLens Frontend" /D "%FRONTEND_DIR%" cmd /k "npm run dev"

timeout /t 3 /nobreak >nul

echo [3/3] Opening browser at http://localhost:5173...
start http://localhost:5173

echo.
echo TruthLens is up and running!
echo Backend ML Service : http://127.0.0.1:8001/health
echo Frontend Dashboard : http://localhost:5173
echo.
pause
