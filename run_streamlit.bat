@echo off
title VisionScayl - Streamlit Web App (GPU Accelerated)
echo ========================================================
echo   VisionScayl - AI-Powered Image Super-Resolution
echo   Launching with .visionscayl (CUDA GPU Acceleration)
echo ========================================================
echo.

cd /d "%~dp0"

REM Activate .visionscayl virtual environment if present
if exist ".visionscayl\Scripts\activate.bat" (
    echo [INFO] Activating .visionscayl virtual environment...
    call .visionscayl\Scripts\activate.bat
    python -c "import torch; print('[GPU INFO] CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    streamlit run streamlit_app.py
) else (
    echo [WARNING] .visionscayl virtual environment not found. Falling back to system Python...
    streamlit run streamlit_app.py
)

pause
