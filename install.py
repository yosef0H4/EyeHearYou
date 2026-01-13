#!/usr/bin/env python3
"""
Installation script for EyeHearYou
Replaces install-gpu.bat and install-cpu.bat with a cross-platform Python script

Usage:
    python install.py [--cpu] [--gpu]

If no option is specified, it will detect GPU availability and install accordingly.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("WARNING: huggingface-hub not available, will use urllib fallback")


def run_command(cmd, check=True, shell=False):
    """Run a command and return success status with real-time output"""
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    try:
        # Run with real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            shell=shell
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        # Wait for process to complete
        returncode = process.wait()
        
        if returncode != 0:
            print(f"\n{'='*60}")
            print(f"ERROR: Command failed with exit code {returncode}")
            print('='*60)
            if check:
                return False
        else:
            print(f"\n{'='*60}")
            print("Command completed successfully")
            print('='*60)
        
        return returncode == 0
        
    except FileNotFoundError:
        print(f"\n{'='*60}")
        print(f"ERROR: Command not found: {cmd[0]}")
        print("Make sure 'uv' is installed and in your PATH")
        print('='*60)
        return False
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Exception while running command: {e}")
        print('='*60)
        return False


def check_uv():
    """Check if uv is available"""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: 'uv' is not installed or not in PATH")
        print("Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def check_cuda():
    """Check if CUDA is available"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_pytorch_gpu():
    """Install PyTorch 2.8.0 with CUDA 12.9"""
    print("\n" + "="*60)
    print("[1/5] Installing PyTorch 2.8.0 with CUDA 12.9...")
    print("="*60)
    print("NOTE: torchvision 0.23.0 is required for PyTorch 2.8.0")
    print("      PyTorch 2.8.0 does NOT require the torchvision patch (unlike 2.7.0)")
    
    cmd = [
        "uv", "pip", "install",
        "torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0",
        "--index-url", "https://download.pytorch.org/whl/cu129"
    ]
    return run_command(cmd)


def install_pytorch_cpu():
    """Install PyTorch CPU version"""
    print("\n" + "="*60)
    print("[1/4] Installing PyTorch CPU version...")
    print("="*60)
    
    cmd = ["uv", "pip", "install", "torch", "torchvision", "torchaudio"]
    return run_command(cmd)


def install_flash_attention():
    """Download and install Flash Attention 2.8.2 using huggingface_hub"""
    print("\n" + "="*60)
    print("[2/5] Installing Flash Attention 2.8.2 (Windows prebuilt wheel)...")
    print("="*60)
    
    if sys.platform != "win32":
        print("WARNING: Flash Attention prebuilt wheel is only available for Windows")
        print("         Skipping Flash Attention installation")
        return True
    
    wheel_name = "flash_attn-2.8.2+cu129torch2.8.0cxx11abiTRUE-cp312-cp312-win_amd64.whl"
    repo_id = "ussoewwin/Flash-Attention-2_for_Windows"
    
    temp_dir = Path(tempfile.gettempdir())
    wheel_path = temp_dir / wheel_name
    
    try:
        if HF_HUB_AVAILABLE:
            print(f"Downloading Flash Attention wheel from Hugging Face Hub...")
            print(f"Repository: {repo_id}")
            print(f"File: {wheel_name}")
            print("(Using huggingface-hub with XET support for efficient download)")
            
            # Use hf_hub_download which automatically uses XET if available
            # hf_hub_download returns the full path to the cached file
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=wheel_name,
                cache_dir=str(temp_dir),
                force_download=False,  # Use cache if available
            )
            
            print(f"Downloaded to: {downloaded_path}")
            
            # hf_hub_download may return a symlink that resolves to a blob file
            # Copy to a temp location with the proper .whl filename for installation
            downloaded_file = Path(downloaded_path)
            if not downloaded_file.exists():
                raise FileNotFoundError(f"Downloaded file not found at: {downloaded_file}")
            
            # Copy to temp_dir with the original filename for installation
            # This ensures uv pip install can recognize it as a wheel file
            wheel_path = temp_dir / wheel_name
            print(f"Copying to installation location: {wheel_path}")
            shutil.copy2(downloaded_file, wheel_path)
            
            print("Download complete!")
        else:
            # Fallback to urllib if huggingface_hub not available
            import urllib.request
            url = f"https://huggingface.co/{repo_id}/resolve/main/{wheel_name}"
            print(f"Downloading Flash Attention wheel from {url}...")
            print(f"Destination: {wheel_path}")
            print("(WARNING: Using urllib fallback - install huggingface-hub for better performance)")
            
            urllib.request.urlretrieve(url, wheel_path)
            print("Download complete!")
        
        print(f"Installing {wheel_name}...")
        print(f"Using file: {wheel_path}")
        
        # Ensure the file exists and use absolute path
        if not wheel_path.exists():
            raise FileNotFoundError(f"Wheel file not found at: {wheel_path}")
        
        install_path_str = str(wheel_path.resolve())
        cmd = ["uv", "pip", "install", install_path_str]
        success = run_command(cmd)
        
        # Clean up the temporary copy (keep the huggingface cache)
        if wheel_path.exists() and wheel_path.parent == temp_dir:
            try:
                wheel_path.unlink()
                print("Cleaned up temporary wheel file")
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")
        
        if HF_HUB_AVAILABLE:
            print("Note: Original file cached by huggingface-hub (will be reused on next install)")
        
        return success
        
    except Exception as e:
        print(f"ERROR: Failed to download/install Flash Attention: {e}")
        if wheel_path.exists():
            try:
                wheel_path.unlink()
            except:
                pass
        return False


def install_h2ovl_deps():
    """Install H2OVL-Mississippi dependencies with exact versions"""
    print("\n" + "="*60)
    print("[3/5] Installing H2OVL-Mississippi dependencies (exact versions)...")
    print("="*60)
    
    cmd = [
        "uv", "pip", "install",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "timm==1.0.24",
        "peft==0.18.1",
        "mss==10.0.0"
    ]
    return run_command(cmd)


def install_kokoro_tts():
    """Install Kokoro TTS dependencies and pre-download default voice"""
    print("\n" + "="*60)
    print("[4/5] Installing Kokoro TTS dependencies...")
    print("="*60)
    print("NOTE: Installing kokoro, misaki[en], loguru, soundfile, and sounddevice")
    print("      These require torch, so they must be installed after PyTorch.")
    print("      Will also pre-download default voice (af_heart) for offline use.")
    
    # Install kokoro and its dependencies with exact versions
    # misaki[en] includes spacy, spacy-curated-transformers, espeakng-loader, num2words, phonemizer-fork
    cmd = [
        "uv", "pip", "install",
        "kokoro==0.9.4",
        "misaki[en]==0.9.4",
        "loguru==0.7.3",
        "soundfile==0.13.1",
        "sounddevice==0.5.3"
    ]
    if not run_command(cmd):
        return False
    
    # Pre-install spacy model to avoid runtime download
    print("\n[4b/5] Pre-installing spacy model (en-core-web-sm) to avoid runtime download...")
    print("NOTE: This model is required by misaki[en] for English G2P")
    cmd = ["uv", "run", "python", "-m", "spacy", "download", "en_core_web_sm"]
    result = run_command(cmd, check=False)
    if not result:
        print("WARNING: Failed to pre-install spacy model - it will be downloaded on first use")
    
    # Pre-download default voice to avoid runtime download (works offline after install)
    print("\n[4c/5] Pre-downloading default voice (af_heart) to avoid runtime download...")
    print("NOTE: This voice file will be cached so the program works offline")
    if not preload_default_voice():
        print("WARNING: Failed to pre-download default voice - it will be downloaded on first use")
    
    return True


def preload_default_voice():
    """Pre-download the default voice (af_heart) from HuggingFace"""
    try:
        # Use uv run python to ensure we're in the correct environment
        preload_script = """
import sys
try:
    from huggingface_hub import hf_hub_download
    repo_id = 'hexgrad/Kokoro-82M'
    voice_file = 'voices/af_heart.pt'
    print(f"Downloading default voice from {repo_id}...")
    print(f"Voice file: {voice_file}")
    print("(This will be cached for offline use)")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=voice_file,
        repo_type='model',
        force_download=False  # Use cache if available
    )
    print(f"✓ Default voice downloaded and cached: {downloaded_path}")
    sys.exit(0)
except ImportError:
    print("ERROR: huggingface-hub not available")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to download default voice: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        cmd = ["uv", "run", "python", "-c", preload_script]
        result = run_command(cmd, check=False)
        return result
    except Exception as e:
        print(f"ERROR: Exception while pre-downloading voice: {e}")
        return False


def install_rapidocr(use_gpu=False):
    """Install RapidOCR and optionally onnxruntime-gpu for GPU acceleration"""
    print("\n" + "="*60)
    print("[5/5] Installing RapidOCR...")
    print("="*60)
    
    # Always install rapidocr-onnxruntime
    cmd = ["uv", "pip", "install", "rapidocr-onnxruntime"]
    success = run_command(cmd)
    
    # Optionally install onnxruntime-gpu if GPU is available
    if use_gpu:
        print("\n[5b/5] Installing onnxruntime-gpu for GPU acceleration...")
        print("NOTE: This enables GPU acceleration for RapidOCR (~27% faster detection)")
        cmd_gpu = ["uv", "pip", "install", "onnxruntime-gpu"]
        gpu_success = run_command(cmd_gpu, check=False)  # Don't fail if GPU install fails
        if not gpu_success:
            print("WARNING: Failed to install onnxruntime-gpu - RapidOCR will use CPU")
            print("         GPU acceleration will not be available")
        else:
            print("✓ onnxruntime-gpu installed successfully")
    
    return success


def verify_installation(gpu=False):
    """Verify the installation using uv run python"""
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60)
    
    # Use uv run python to verify in the correct environment
    verify_script = f"""
import sys
checks = []

# Check PyTorch
try:
    import torch
    version = torch.__version__
    cuda_available = torch.cuda.is_available() if {gpu} else None
    if {gpu}:
        status = f"PyTorch: {{version}}, CUDA: {{cuda_available}}"
        checks.append(("PyTorch (GPU)", cuda_available))
    else:
        status = f"PyTorch: {{version}}"
        checks.append(("PyTorch (CPU)", True))
    print(f"[OK] {{status}}")
except ImportError:
    print("[FAIL] PyTorch: Not installed")
    checks.append(("PyTorch", False))

# Check RapidOCR
try:
    from rapidocr_onnxruntime import RapidOCR
    print("[OK] RapidOCR: Installed successfully")
    checks.append(("RapidOCR", True))
except ImportError as e:
    error_msg = str(e)
    print(f"[FAIL] RapidOCR: Not installed or missing dependencies: " + error_msg)
    checks.append(("RapidOCR", False))

# Check Kokoro TTS
try:
    from kokoro import KPipeline
    print("[OK] Kokoro TTS: Installed successfully")
    checks.append(("Kokoro TTS", True))
except ImportError:
    print("[FAIL] Kokoro TTS: Not installed")
    checks.append(("Kokoro TTS", False))

if {gpu}:
    # Check Flash Attention
    try:
        import flash_attn
        print("[OK] Flash Attention: Installed successfully")
        checks.append(("Flash Attention", True))
    except ImportError:
        print("[WARN] Flash Attention: Not installed (optional)")
        checks.append(("Flash Attention", None))

all_passed = all(status for _, status in checks if status is not None)
if all_passed:
    print("\\nInstallation verification: SUCCESS")
    sys.exit(0)
else:
    print("\\nInstallation verification: SOME CHECKS FAILED")
    print("\\nFailed checks:")
    for name, status in checks:
        if status is False:
            print(f"  - {{name}}")
    sys.exit(1)
"""
    
    cmd = ["uv", "run", "python", "-c", verify_script]
    result = run_command(cmd, check=False)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Install dependencies for EyeHearYou",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py          # Auto-detect GPU and install accordingly
  python install.py --gpu    # Force GPU installation
  python install.py --cpu    # Force CPU installation
        """
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Install CPU-only version (no CUDA support)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Install GPU version with CUDA support"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("EyeHearYou - Installation Script")
    print("="*60)
    print()
    
    # Check prerequisites
    if not check_uv():
        sys.exit(1)
    
    # Determine installation type
    if args.cpu and args.gpu:
        print("ERROR: Cannot specify both --cpu and --gpu")
        sys.exit(1)
    
    if args.cpu:
        use_gpu = False
        print("\n" + "="*60)
        print("⚠️  WARNING: CPU-only installation requested")
        print("="*60)
        print("⚠️  IMPORTANT: This project is designed for NVIDIA GPU with CUDA.")
        print("⚠️  CPU implementation is UNTESTED and will be VERY SLOW.")
        print("⚠️  Transformer models (H2OVL-Mississippi, Kokoro TTS) are heavy")
        print("⚠️  and may take 10-30+ seconds per capture on CPU.")
        print("⚠️  GPU is STRONGLY RECOMMENDED for acceptable performance.")
        print("="*60)
        response = input("\nContinue with CPU installation? (y/N): ").strip().lower()
        if response != 'y':
            print("Installation cancelled. Please install CUDA and an NVIDIA GPU.")
            sys.exit(0)
        print("\nProceeding with CPU-only installation (not recommended)...")
    elif args.gpu:
        use_gpu = True
        print("Installing GPU version with CUDA support...")
    else:
        # Auto-detect
        use_gpu = check_cuda()
        if use_gpu:
            print("CUDA detected. Installing GPU version...")
        else:
            print("\n" + "="*60)
            print("⚠️  WARNING: No CUDA detected. Installing CPU version...")
            print("="*60)
            print("⚠️  IMPORTANT: This project is designed for NVIDIA GPU with CUDA.")
            print("⚠️  CPU implementation is UNTESTED and will be VERY SLOW.")
            print("⚠️  Transformer models (H2OVL-Mississippi, Kokoro TTS) are heavy")
            print("⚠️  and may take 10-30+ seconds per capture on CPU.")
            print("⚠️  GPU is STRONGLY RECOMMENDED for acceptable performance.")
            print("="*60)
            response = input("\nContinue with CPU installation? (y/N): ").strip().lower()
            if response != 'y':
                print("Installation cancelled. Please install CUDA and an NVIDIA GPU.")
                sys.exit(0)
            print("\nProceeding with CPU installation (not recommended)...")
            print("(Use --gpu to force GPU installation anyway)")
    
    print()
    
    # Ensure base dependencies are installed first (from pyproject.toml)
    # This includes opencv-python which RapidOCR requires
    print("\n" + "="*60)
    print("[0/5] Ensuring base dependencies are installed...")
    print("="*60)
    print("NOTE: Running 'uv sync' to install dependencies from pyproject.toml")
    print("      (This includes opencv-python, which RapidOCR requires)")
    sync_result = run_command(["uv", "sync"], check=False)
    if not sync_result:
        print("WARNING: uv sync had issues, but continuing with installation...")
        print("         If RapidOCR verification fails, run 'uv sync' manually")
    print()
    
    # Run installation
    success = True
    
    if use_gpu:
        # GPU installation steps
        success = install_pytorch_gpu() and success
        success = install_flash_attention() and success
        success = install_h2ovl_deps() and success
        success = install_kokoro_tts() and success
        success = install_rapidocr(use_gpu=use_gpu) and success
    else:
        # CPU installation steps
        success = install_pytorch_cpu() and success
        success = install_h2ovl_deps() and success
        success = install_kokoro_tts() and success
        success = install_rapidocr(use_gpu=use_gpu) and success
    
    if not success:
        print("\n" + "="*60)
        print("ERROR: Installation failed at one or more steps")
        print("="*60)
        sys.exit(1)
    
    # Ensure opencv-python is installed (it may have been removed during installation)
    # RapidOCR requires cv2 (opencv-python)
    print("\n" + "="*60)
    print("[Final] Ensuring opencv-python is installed (required for RapidOCR)...")
    print("="*60)
    print("NOTE: opencv-python may have been removed during dependency resolution.")
    print("      Reinstalling exact version to ensure RapidOCR works correctly.")
    run_command(["uv", "pip", "install", "opencv-python==4.11.0.86"], check=False)
    print()
    
    # Verify installation
    verify_installation(gpu=use_gpu)
    
    print("\n" + "="*60)
    print("Installation Complete!")
    print("="*60)
    if use_gpu:
        print("\nNOTE: PyTorch 2.8.0 does NOT require the torchvision patch (unlike 2.7.0).")
        print("      The patch is still available for backward compatibility if needed.")
    print("\nYou can now run the application:")
    print("  uv run python run_gui.py")
    print()


if __name__ == "__main__":
    main()

