@echo off
echo ========================================
echo   GamePlan AI - Starting All Servers
echo ========================================
echo.

REM Change to project root directory
cd /d "%~dp0"

echo [1/4] Starting Authentication API Server (Port 5002)...
start "Auth API Server" cmd /k "cd /d %~dp0database && python auth_api.py"
timeout /t 3 /nobreak >nul

echo [2/4] Starting Free Kick API Server (Port 5001)...
start "Free Kick API Server" cmd /k "cd /d %~dp0backend\hassaa\data && python freekick_api.py"
timeout /t 3 /nobreak >nul

echo [3/4] Starting Corner Kick API Server (Port 5000)...
start "Corner Kick API Server" cmd /k "cd /d %~dp0backend\hassaa\data && python api_server.py"
timeout /t 5 /nobreak >nul

echo [4/4] Starting React Frontend (Port 3000)...
start "React Frontend" cmd /k "cd /d %~dp0frontend && npm start"

echo.
echo ========================================
echo   All servers are starting...
echo ========================================
echo.
echo API Servers:
echo   - Auth API:      http://localhost:5002
echo   - Free Kick API: http://localhost:5001
echo   - Corner API:    http://localhost:5000
echo.
echo Frontend:
echo   - React App:     http://localhost:3000
echo.
echo Press any key to exit (servers will continue running)...
pause >nul

