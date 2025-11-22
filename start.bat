@echo off
REM Activate the Python virtual environment
call "E:\Users\Tibo\Obsidian\PNJisme\PNJisme\LAMATRON\ttrpg_env\Scripts\activate.bat"

REM Run the TTRPG agent
python "E:\Users\Tibo\Obsidian\PNJisme\PNJisme\LAMATRON\ollama_ttrpg_agent.py"

REM Pause so you can see any messages when it exits
pause
