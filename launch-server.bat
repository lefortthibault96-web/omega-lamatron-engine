@echo off
REM Se placer dans le dossier du script
cd /d "%~dp0"
call .venv\Scripts\activate

REM Activer le venv
call uvicorn app.main:app --reload --host 0.0.0.0 --port 8008

REM garder la fenêtre ouverte quand c'est fini
pause

