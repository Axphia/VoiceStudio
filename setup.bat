@echo off
REM VoiceStudio Initial Setup Script
echo =======================================
echo 🎙️  VoiceStudio Setup
echo =======================================
echo.

echo [1/3] Creating necessary folders...
if not exist "data\videos" mkdir "data\videos"
if not exist "data\audio_clean" mkdir "data\audio_clean"
if not exist "data\voices" mkdir "data\voices"
if not exist "data\ft" mkdir "data\ft"
if not exist "models" mkdir "models"
if not exist "outputs" mkdir "outputs"
if not exist "training" mkdir "training"
echo ✅ Folders created.
echo.

echo [2/3] Checking uv package manager...
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ uv is not installed!
    echo Please install uv from: https://github.com/astral-sh/uv
    echo Or run: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)
echo ✅ uv is installed.
echo.

echo [3/3] Installing Python dependencies (this may take a while)...
uv sync
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies.
    pause
    exit /b 1
)
echo ✅ Dependencies installed.
echo.

echo =======================================
echo 🎉 Setup Complete! 
echo =======================================
echo You can now put your videos in 'data\videos' 
echo and double-click 'start.bat' or 'colab_prep.bat'.
echo.
pause