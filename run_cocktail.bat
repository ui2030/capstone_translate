@echo off
setlocal
cd /d "%~dp0"

echo [Cocktail] Environment quick check
python --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found. Install Python 3.10+ or use the packaged exe build.
    pause
    exit /b 1
)

python diagnose_environment.py
if errorlevel 1 (
    echo.
    echo [WARN] Diagnostics reported a problem. The app may still run with limited features.
    echo.
)

for %%F in (Cocktail*.py) do (
    set APP_FILE=%%F
    goto :run
)

echo [ERROR] Cocktail Python entrypoint was not found.
pause
exit /b 1

:run
echo [Cocktail] Starting %APP_FILE%
python "%APP_FILE%"
if errorlevel 1 (
    echo.
    echo [ERROR] Cocktail exited with an error.
    pause
)

