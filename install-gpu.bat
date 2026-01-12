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
REM - Flash Attention 2 (Windows prebuilt wheel)
REM - H2OVL-Mississippi dependencies
REM - PaddlePaddle GPU (you may need to adjust CUDA version)
REM - PaddleOCR

echo ========================================
echo GPU Requirements Installation
echo ========================================
echo.

echo [1/5] Installing PyTorch 2.7.0 with CUDA 12.8...
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo [2/5] Installing Flash Attention 2 (Windows prebuilt wheel)...
uv pip install "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl"
if errorlevel 1 (
    echo ERROR: Failed to install Flash Attention
    pause
    exit /b 1
)

echo.
echo [3/5] Installing H2OVL-Mississippi dependencies...
uv pip install transformers accelerate timm peft
if errorlevel 1 (
    echo ERROR: Failed to install H2OVL dependencies
    pause
    exit /b 1
)

echo.
echo [4/5] Installing PaddlePaddle GPU...
echo NOTE: This script uses CUDA 12.9. If you have a different CUDA version, edit this script.
echo Available versions: cu129, cu128, cu121, cu118, or cu102 (from PyPI)
uv pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu129/
if errorlevel 1 (
    echo WARNING: Failed to install PaddlePaddle GPU. Trying alternative...
    uv pip install paddlepaddle-gpu==2.6.2
)

echo.
echo [5/5] Installing PaddleOCR...
uv pip install paddleocr
if errorlevel 1 (
    echo ERROR: Failed to install PaddleOCR
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
echo   python -c "import paddle; print(f'PaddlePaddle: {paddle.__version__}, Device: {paddle.device.get_device()}')"
echo.
pause

