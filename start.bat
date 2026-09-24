@echo off
REM VoiceStudio launcher - double-click, browser opens, no commands needed.
cd /d "%~dp0"
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python scripts/api.py --port 9999
start "http://127.0.0.1:9999"
pause
