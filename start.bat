@echo off
REM Move to the folder containing this batch file
cd /d "%~dp0"
REM Name of the virtual environment
set "VENV_NAME=ttrpg_env"
set "VENV_DIR=%CD%\%VENV_NAME%"

REM Create venv if missing
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Virtual environment not found. Creating %VENV_NAME%...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERREUR] Echec de creation du virtual environment.
        pause
        exit /b 1
    )
)
REM Activate the virtual environment
call ttrpg_env\Scripts\activate
REM Install requirement
pip install -r requirements.txt
REM Run the TTRPG agent using the venv Python
python ollama_ttrpg_agent.py

REM Pause so you can see messages when it exits

pause

