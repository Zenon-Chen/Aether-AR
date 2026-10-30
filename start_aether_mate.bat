@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ========================================================
echo   Aether-AR - Quick Bootstrap ^& Launcher
echo ========================================================

if not exist ".venv\Scripts\python.exe" (
    echo [Setup] Creating Python virtual environment and installing dependencies...
    where uv >nul 2>nul
    if %errorlevel%==0 (
        uv venv --python 3.11
        uv pip install -e ".[edge]"
    ) else (
        python -m venv .venv
        .venv\Scripts\python.exe -m pip install --upgrade pip
        .venv\Scripts\python.exe -m pip install -e ".[edge]"
    )
)

.venv\Scripts\python.exe launcher.py %*
