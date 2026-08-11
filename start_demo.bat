@echo off
cd /d "%~dp0"
echo ===============================================
echo  Starting Drugenix (backend + frontend)
echo ===============================================

echo Starting backend on http://127.0.0.1:8000 ...
start "Drugenix Backend" cmd /k "python scripts\run_api.py"

timeout /t 3 /nobreak >nul

echo Starting frontend on http://localhost:5174 ...
cd frontend
start "Drugenix Frontend" cmd /k "npm run dev"
cd ..

timeout /t 5 /nobreak >nul

echo Opening browser...
start http://localhost:5174

echo.
echo Two windows are now running - do NOT close them during your demo.
echo   - "Drugenix Backend"  (API on port 8000)
echo   - "Drugenix Frontend" (UI on port 5174)
echo.
echo To stop everything, just close both of those windows.
pause
