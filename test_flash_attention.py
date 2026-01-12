#!/usr/bin/env python3
"""
Test script to find the latest working Flash Attention version
Tests different PyTorch + Flash Attention combinations

Usage:
    python test_flash_attention.py --version 2.8.3 --pytorch 2.9.1 --cuda 13.0 --source ussoewwin
    python test_flash_attention.py --version 2.8.3 --pytorch 2.9.1 --cuda 13.0 --source ussoewwin --no-patch
"""

import argparse
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("WARNING: huggingface-hub not available")


def run_command(cmd, check=True, shell=False):
    """Run a command and return success status with real-time output (like install.py)"""
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
        print('='*60)
        return False
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Exception while running command: {e}")
        print('='*60)
        return False


def run_test(script_name, no_patch=False):
    """Run a test script and return success status with real-time output"""
    cmd = ["uv", "run", "python", script_name]
    if no_patch and script_name == "test_h2ovl.py":
        cmd.append("--no-patch")
    if no_patch and script_name == "test_kokoro_tts.py":
        cmd.append("--no-patch")
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    
    try:
        # Run with real-time output streaming (like install.py)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        # Wait for process to complete
        returncode = process.wait()
        
        if returncode == 0:
            print(f"\n{'='*60}")
            print(f"[OK] {script_name} passed")
            print('='*60)
            return True
        else:
            print(f"\n{'='*60}")
            print(f"[FAIL] {script_name} failed with exit code {returncode}")
            print('='*60)
            return False
    except subprocess.TimeoutExpired:
        print(f"\n{'='*60}")
        print(f"[FAIL] {script_name} timed out")
        print('='*60)
        return False
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"[FAIL] {script_name} error: {e}")
        print('='*60)
        return False


def test_flash_attention_version(pytorch_version, cuda_version, flash_attn_version, source, no_patch=False):
    """Test a specific Flash Attention version"""
    print("\n" + "="*60)
    print(f"Testing Flash Attention {flash_attn_version}")
    print(f"  PyTorch: {pytorch_version}")
    print(f"  CUDA: {cuda_version}")
    print(f"  Source: {source}")
    print(f"  Patch: {'DISABLED' if no_patch else 'ENABLED'}")
    print("="*60)
    
    # Determine wheel filename based on source
    if source == "ussoewwin":
        # Format: flash_attn-{version}+cu{cuda}torch{pytorch}cxx11abiTRUE-cp312-cp312-win_amd64.whl
        # Example: flash_attn-2.8.2+cu129torch2.8.0cxx11abiTRUE-cp312-cp312-win_amd64.whl
        cuda_short = cuda_version.replace(".", "")
        wheel_name = f"flash_attn-{flash_attn_version}+cu{cuda_short}torch{pytorch_version}cxx11abiTRUE-cp312-cp312-win_amd64.whl"
        repo_id = "ussoewwin/Flash-Attention-2_for_Windows"
    elif source == "lldacing":
        # Format: flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl
        cuda_short = cuda_version.replace(".", "")
        # Check if PyTorch version >= 2.8.0 for ABI
        pytorch_major_minor = tuple(map(int, pytorch_version.split(".")[:2]))
        abi = "TRUE" if pytorch_major_minor >= (2, 8) else "FALSE"
        wheel_name = f"flash_attn-{flash_attn_version}+cu{cuda_short}torch{pytorch_version}cxx11abi{abi}-cp312-cp312-win_amd64.whl"
        repo_id = "lldacing/flash-attention-windows-wheel"
    else:
        print(f"[ERROR] Unknown source: {source}")
        return False
    
    print(f"\nWheel: {wheel_name}")
    print(f"Repository: {repo_id}")
    
    # Uninstall current versions
    print("\n" + "="*60)
    print("[1/5] Uninstalling current PyTorch and Flash Attention...")
    print("="*60)
    # uv pip uninstall doesn't support -y, packages are uninstalled interactively or we skip
    # Just try to uninstall, user may need to confirm
    run_command(["uv", "pip", "uninstall", "torch", "torchvision", "torchaudio", "flash-attn"], check=False)
    
    # Install PyTorch
    print(f"\n[2/5] Installing PyTorch {pytorch_version} with CUDA {cuda_version}...")
    
    # Map CUDA versions to PyTorch index URLs
    cuda_mapping = {
        "13.0": "cu131",
        "12.9": "cu129", 
        "12.8": "cu128",
        "12.4": "cu124",
    }
    
    # Get CUDA short version
    cuda_short = None
    for cuda_full, short in cuda_mapping.items():
        if cuda_version.startswith(cuda_full.split(".")[0] + "." + cuda_full.split(".")[1]):
            cuda_short = short
            break
    
    if not cuda_short:
        # Try to construct from version
        cuda_short = f"cu{cuda_version.replace('.', '')}"
    
    index_url = f"https://download.pytorch.org/whl/{cuda_short}"
    
    # Get compatible torchvision version
    pytorch_major_minor = ".".join(pytorch_version.split(".")[:2])
    
    # Map PyTorch versions to compatible torchvision versions
    # PyTorch 2.9.x -> torchvision 0.24.x (if available)
    # PyTorch 2.8.x -> torchvision 0.23.x
    # PyTorch 2.7.x -> torchvision 0.22.x
    torchvision_map = {
        "2.9": "0.24.0",
        "2.8": "0.23.0", 
        "2.7": "0.22.0",
    }
    tv_version = torchvision_map.get(pytorch_major_minor, pytorch_version)
    
    # Try installing PyTorch
    print(f"Attempting: torch=={pytorch_version} from {index_url}")
    cmd = ["uv", "pip", "install", f"torch=={pytorch_version}", 
           f"torchvision=={tv_version}", f"torchaudio=={pytorch_version}",
           "--index-url", index_url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # If that fails, try just torch first
    if result.returncode != 0:
        print(f"First attempt failed, trying torch only...")
        cmd = ["uv", "pip", "install", f"torch=={pytorch_version}", 
               "--index-url", index_url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Install compatible torchvision and torchaudio
            cmd = ["uv", "pip", "install", f"torchvision=={tv_version}", 
                   f"torchaudio=={pytorch_version}", "--index-url", index_url]
            result = subprocess.run(cmd, capture_output=True, text=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAIL] Failed to install PyTorch: {result.stderr}")
        return False
    print("[OK] PyTorch installed")
    
    # Download and install Flash Attention
    print("\n" + "="*60)
    print(f"[3/5] Downloading Flash Attention from {repo_id}...")
    print("="*60)
    try:
        from huggingface_hub import hf_hub_download
        import tempfile
        import shutil
        from pathlib import Path as PPath
        
        temp_dir = PPath(tempfile.gettempdir())
        print(f"Downloading: {wheel_name}")
        print(f"Repository: {repo_id}")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=wheel_name,
            cache_dir=str(temp_dir),
            force_download=False,
        )
        
        print(f"Downloaded to: {downloaded_path}")
        
        # Copy to temp location with proper filename
        wheel_path = temp_dir / wheel_name
        print(f"Copying to: {wheel_path}")
        shutil.copy2(PPath(downloaded_path), wheel_path)
        
        print("\n" + "="*60)
        print(f"[4/5] Installing Flash Attention wheel...")
        print("="*60)
        cmd = ["uv", "pip", "install", str(wheel_path.resolve())]
        if not run_command(cmd):
            print(f"[FAIL] Failed to install Flash Attention")
            return False
        print("[OK] Flash Attention installed")
        
        # Clean up
        if wheel_path.exists():
            try:
                wheel_path.unlink()
                print("Cleaned up temporary wheel file")
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")
            
    except Exception as e:
        print(f"[FAIL] Failed to download/install Flash Attention: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run tests
    print(f"\n[5/5] Running tests...")
    
    kokoro_ok = run_test("test_kokoro_tts.py", no_patch=no_patch)
    h2ovl_ok = run_test("test_h2ovl.py", no_patch=no_patch)
    
    success = kokoro_ok and h2ovl_ok
    
    print("\n" + "="*60)
    if success:
        print(f"[SUCCESS] Flash Attention {flash_attn_version} with PyTorch {pytorch_version} works!")
    else:
        print(f"[FAIL] Flash Attention {flash_attn_version} with PyTorch {pytorch_version} failed")
        print(f"  Kokoro TTS: {'PASS' if kokoro_ok else 'FAIL'}")
        print(f"  H2OVL: {'PASS' if h2ovl_ok else 'FAIL'}")
    print("="*60)
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Test Flash Attention versions")
    parser.add_argument("--version", required=True, help="Flash Attention version (e.g., 2.8.3)")
    parser.add_argument("--pytorch", required=True, help="PyTorch version (e.g., 2.9.1)")
    parser.add_argument("--cuda", required=True, help="CUDA version (e.g., 13.0)")
    parser.add_argument("--source", default="ussoewwin", choices=["ussoewwin", "lldacing"],
                       help="Source repository (default: ussoewwin)")
    parser.add_argument("--no-patch", action="store_true", 
                       help="Test without torchvision patch")
    
    args = parser.parse_args()
    
    success = test_flash_attention_version(
        args.pytorch, args.cuda, args.version, args.source, args.no_patch
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

