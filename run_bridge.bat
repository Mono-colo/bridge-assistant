@echo off
cd /d "%~dp0"
python -m bridge_assistant
if errorlevel 1 pause

