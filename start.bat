@echo off
REM Se placer dans le dossier du script
cd /d "%~dp0"

REM Activer le venv
call ttrpg_env\Scripts\activate

REM Run the TTRPG agent
py ollama_ttrpg_agent.py

REM Pause so you can see any messages when it exits
pause