@echo off
REM VoiceStudio Colab Prep — double-click to open the training preparation app
cd /d "%~dp0"
start "" "http://127.0.0.1:9998"
uv run python scripts/colab_prep.py
pause
