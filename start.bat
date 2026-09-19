@echo off
REM VoiceStudio launcher - double-click, browser opens, no commands needed.
cd /d "%~dp0"
start "" "http://127.0.0.1:8000"
uv run python scripts/api.py --port 8000
pause
