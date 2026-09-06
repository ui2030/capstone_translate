@echo off
rem Cocktail launcher.
rem ASCII ONLY in this file. Non-ASCII text (Hangul comments, Hangul paths) is
rem decoded with the console OEM codepage (CP949) and breaks under UTF-8 shells.
rem Korean setup guide: see ENVIRONMENT.md
setlocal
cd /d "%~dp0"

set "PYEXE="
set "PYARGS="

rem Interpreter search order. First one that works wins.
rem   1) COCKTAIL_PYTHON     - explicit override, always honored if it runs
rem   2) anaconda/miniconda "cd" env - the environment this app is developed in
rem   3) py -3.11            - Windows Python launcher
rem   4) python              - whatever is on PATH
if defined COCKTAIL_PYTHON call :pick_forced "%COCKTAIL_PYTHON%"
if not defined PYEXE call :pick "%USERPROFILE%\anaconda3\envs\cd\python.exe"
if not defined PYEXE call :pick "%USERPROFILE%\miniconda3\envs\cd\python.exe"
if not defined PYEXE call :pick_py
if not defined PYEXE call :pick "python"

if not defined PYEXE (
    echo [ERROR] No usable Python was found.
    echo         Cocktail needs Python 3.10 / 3.11 with the packages in requirements.txt.
    echo.
    echo   Fix 1: install the dependencies
    echo            pip install -r requirements.txt
    echo   Fix 2: point Cocktail at the right interpreter, then run this file again
    echo            set COCKTAIL_PYTHON=C:\path\to\python.exe
    echo.
    echo   Setup guide ^(Korean^): ENVIRONMENT.md
    echo.
    pause
    exit /b 1
)

echo [Cocktail] Python: %PYEXE% %PYARGS%
echo [Cocktail] Environment quick check
"%PYEXE%" %PYARGS% diagnose_environment.py
if errorlevel 1 (
    echo.
    echo [WARN] Diagnostics reported a problem. The app may still run with limited features.
    echo.
)

rem Entrypoint is plain ASCII now (was Cocktail<hangul>.py, matched by glob).
set APP_FILE=cocktail.py
if not exist "%APP_FILE%" (
    echo [ERROR] Cocktail Python entrypoint was not found: %APP_FILE%
    pause
    exit /b 1
)

:run
echo [Cocktail] Starting %APP_FILE%
"%PYEXE%" %PYARGS% "%APP_FILE%"
if errorlevel 1 (
    echo.
    echo [ERROR] Cocktail exited with an error.
    pause
)
exit /b 0

rem --- helpers -------------------------------------------------------------
rem pick_forced: user said which interpreter to use. Only check that it runs;
rem missing packages are reported by diagnose_environment.py, not hidden here.
:pick_forced
"%~1" -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [WARN] COCKTAIL_PYTHON is set but does not run: %~1
    goto :eof
)
set "PYEXE=%~1"
goto :eof

rem pick: candidate must be able to import PySide6 - that is what separates a
rem usable interpreter from a bare system Python with no packages installed.
:pick
"%~1" -c "import PySide6" >nul 2>nul
if errorlevel 1 goto :eof
set "PYEXE=%~1"
goto :eof

:pick_py
py -3.11 -c "import PySide6" >nul 2>nul
if errorlevel 1 goto :eof
set "PYEXE=py"
set "PYARGS=-3.11"
goto :eof
