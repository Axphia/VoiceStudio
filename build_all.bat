@echo off
echo ==========================================
echo  Building all VoiceStudio Rust Executables
echo ==========================================
cargo build --release --workspace
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b %errorlevel%
)
if not exist "bin" mkdir bin
copy /Y target\release\voice-prep.exe bin\
copy /Y target\release\voice-server.exe bin\
copy /Y target\release\voice-burn.exe bin\
echo ==========================================
echo  All .EXE files successfully built to bin\
echo   - bin\voice-prep.exe
echo   - bin\voice-server.exe
echo   - bin\voice-burn.exe
echo ==========================================
pause
