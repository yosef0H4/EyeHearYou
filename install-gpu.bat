@echo off
REM GPU Installation Script for Visual Novel OCR
REM For users with NVIDIA GPU and CUDA support
REM
REM Prerequisites:
REM 1. Run 'uv sync' first to install base dependencies
REM 2. Check your CUDA version: nvidia-smi
REM
REM This script installs:
REM - PyTorch 2.7.0 with CUDA 12.8
REM - Flash Attention 2.7.4 (Windows prebuilt wheel for torch 2.7.0 + CUDA 12.8)
REM - H2OVL-Mississippi dependencies (transformers, accelerate, timm, peft)
REM - Kokoro TTS dependencies (kokoro, soundfile, sounddevice)
REM - RapidOCR (CPU via ONNX Runtime, no GPU needed)
REM
REM NOTE: This configuration (PyTorch 2.7.0 + CUDA 12.8 + Flash Attention 2.7.4)
REM is the tested and working setup. Other versions may have compatibility issues.

echo ========================================
echo GPU Requirements Installation
echo ========================================
echo.

echo [1/4] Installing PyTorch 2.7.0 with CUDA 12.8...
echo NOTE: torchvision 0.22.0 is required for PyTorch 2.7.0 (for Flash Attention compatibility)
echo       A runtime patch handles the known compatibility issue automatically.
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo [2/4] Installing Flash Attention 2.7.4 (Windows prebuilt wheel)...
REM Use Python to download (handles URL encoding properly), then install with uv
REM Download with proper filename to avoid uv pip install issues
python -c "import urllib.request, os, shutil; temp_dir = os.environ.get('TEMP', '.'); src = os.path.join(temp_dir, 'flash_attn_temp.whl'); dst = os.path.join(temp_dir, 'flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl'); urllib.request.urlretrieve('https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl', src); shutil.move(src, dst); print(f'Downloaded: {dst}')"
if errorlevel 1 (
    echo ERROR: Failed to download Flash Attention wheel
    pause
    exit /b 1
)
uv pip install "%TEMP%\flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl"
if errorlevel 1 (
    echo ERROR: Failed to install Flash Attention
    del "%TEMP%\flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl" 2>nul
    pause
    exit /b 1
)
del "%TEMP%\flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl" 2>nul

echo.
echo [3/4] Installing H2OVL-Mississippi dependencies...
uv pip install transformers accelerate timm peft
if errorlevel 1 (
    echo ERROR: Failed to install H2OVL dependencies
    pause
    exit /b 1
)

echo.
echo [4/5] Installing Kokoro TTS dependencies...
uv pip install kokoro>=0.9.2 soundfile>=0.12.1 sounddevice>=0.4.6
if errorlevel 1 (
    echo ERROR: Failed to install Kokoro TTS dependencies
    pause
    exit /b 1
)

echo.
echo [5/5] Installing RapidOCR (CPU via ONNX Runtime)...
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
echo   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo   python -c "from rapidocr_onnxruntime import RapidOCR; print('RapidOCR: Installed successfully')"
echo.
pause

