@echo off
setlocal enableextensions enabledelayedexpansion

rem Always run from this script's folder
cd /d "%~dp0"

rem If a local override exists (gitignored), delegate to it for machine-specific paths
if exist run_scraper.local.bat (
    call run_scraper.local.bat %*
    set "ERR=%ERRORLEVEL%"
    echo.
    echo Script finished. Press any key to exit.
    pause
    endlocal & exit /b %ERR%
)

rem Sanitized launcher: prefer PYTHON_EXE, then .venv, then py, then python
set "PY=%PYTHON_EXE%"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    py -3 -V >nul 2>&1 && set "PY=py -3"
)
if not defined PY set "PY=python"

%PY% run_scraper.py %*
set "ERR=%ERRORLEVEL%"

echo.
echo Script finished. Press any key to exit.
pause

endlocal & exit /b %ERR%
