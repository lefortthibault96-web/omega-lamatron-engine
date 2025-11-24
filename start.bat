@echo off
REM Move to the folder containing this batch file
cd /d "%~dp0"

REM Activate the virtual environment
call ttrpg_env\Scripts\activate

REM Run the TTRPG agent using the venv Python
python ollama_ttrpg_agent.py

REM Pause so you can see messages when it exits
pause