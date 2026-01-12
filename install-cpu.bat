@echo off
REM CPU Installation Script for Visual Novel OCR
REM For users without GPU or who want CPU-only installation
REM
REM Prerequisites:
REM 1. Run 'uv sync' first to install base dependencies
REM
REM This script installs:
REM - PyTorch CPU version
REM - H2OVL-Mississippi dependencies
REM - RapidOCR (CPU via ONNX Runtime)

echo ========================================
echo CPU Requirements Installation
echo ========================================
echo.

echo [1/3] Installing PyTorch CPU version...
uv pip install torch torchvision torchaudio
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo [2/3] Installing H2OVL-Mississippi dependencies...
uv pip install transformers accelerate timm peft
if errorlevel 1 (
    echo ERROR: Failed to install H2OVL dependencies
    pause
    exit /b 1
)

echo.
echo [3/3] Installing RapidOCR (CPU via ONNX Runtime)...
uv pip install rapidocr-onnxruntime
if errorlevel 1 (
    echo ERROR: Failed to install RapidOCR
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Verify installation:
echo   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo   python -c "from rapidocr_onnxruntime import RapidOCR; print('RapidOCR: Installed successfully')"
echo.
pause

