@echo off
cd /d "%~dp0"
echo ===============================================
echo  Drugenix - one-time setup
echo ===============================================
echo.
echo Installing backend dependencies (from requirements.txt)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Backend dependency install FAILED. Check your Python/pip installation.
    pause
    exit /b 1
)

echo.
echo Installing frontend dependencies (npm install)...
cd frontend
call npm install
if errorlevel 1 (
    echo.
    echo Frontend dependency install FAILED. Check your Node.js/npm installation.
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo ===============================================
echo  Setup complete. Run start_demo.bat to launch.
echo ===============================================
pause
