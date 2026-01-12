# Flash Attention Version Testing Results

This document tracks testing of different Flash Attention versions with PyTorch to find the latest working combination.

## Test Criteria
- Both `test_kokoro_tts.py` and `test_h2ovl.py` must pass
- Test with and without torchvision patch
- Target: Python 3.12, latest PyTorch + CUDA combination

## Test Results

### Test Script
Run tests with:
```bash
uv run python test_flash_attention.py --version <version> --pytorch <pytorch_version> --cuda <cuda_version> [--no-patch]
```

## Available Flash Attention Sources

### 1. ussoewwin/Flash-Attention-2_for_Windows
**Repository**: https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows

**Available versions for Python 3.12:**
- `flash_attn-2.8.3+cu130torch2.9.1cxx11abiTRUE-cp312-cp312-win_amd64.whl` (PyTorch 2.9.1, CUDA 13.0) - **LATEST**
- `flash_attn-2.8.3+cu130torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl` (PyTorch 2.9.0, CUDA 13.0)
- `flash_attn-2.8.2+cu130torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl` (PyTorch 2.9.0, CUDA 13.0)
- `flash_attn-2.8.2+cu129torch2.8.0cxx11abiTRUE-cp312-cp312-win_amd64.whl` (PyTorch 2.8.0, CUDA 12.9) - **✅ WORKING**

### 2. lldacing/flash-attention-windows-wheel
**Repository**: https://huggingface.co/lldacing/flash-attention-windows-wheel

**Available versions for Python 3.12:**
- `flash_attn-2.7.4.post1+cu128torch2.8.0cxx11abiTRUE-cp312-cp312-win_amd64.whl` (PyTorch 2.8.0, CUDA 12.8) - **NOT FOUND**
- `flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl` (PyTorch 2.7.0, CUDA 12.8) - **CURRENTLY USED**

### 3. petermg/flash_attn_windows (GitHub Releases)
**Repository**: https://github.com/petermg/flash_attn_windows/releases

**Available versions:**
- `flash_attn-2.7.4.post1+cu128.torch270-cp313-cp313-win_amd64.whl` (PyTorch 2.7.0, CUDA 12.8, Python 3.13)
- `flash_attn-2.7.4.post1+cu128.torch270-cp310-cp310-win_amd64.whl` (PyTorch 2.7.0, CUDA 12.8, Python 3.10)

## Testing Order (Latest to Oldest)

**Note**: PyTorch 2.9.x versions are not available yet. Testing order:

1. ✅ **PyTorch 2.8.0 + CUDA 12.9** (ussoewwin, flash_attn 2.8.2) - **LATEST WORKING**
2. ❌ **PyTorch 2.8.0 + CUDA 13.0** (ussoewwin, flash_attn 2.8.3) - PyTorch 2.8.0 not available for CUDA 13.0
3. ❌ **PyTorch 2.8.0 + CUDA 12.8** (lldacing, flash_attn 2.7.4.post1) - Wheel not available
4. ✅ **PyTorch 2.7.0 + CUDA 12.8** (lldacing, flash_attn 2.7.4.post1) - **CURRENTLY USED IN install.py**

## Test Results Log

| PyTorch | CUDA | Flash Attn | Source | Kokoro TTS | H2OVL | Patch Required | Notes |
|---------|------|------------|--------|------------|-------|----------------|-------|
| 2.8.0 | 12.9 | 2.8.2 | ussoewwin | ✅ | ✅ | ❌ **NOT REQUIRED** | **LATEST WORKING** - Tested successfully, patch not needed! |
| 2.7.0 | 12.8 | 2.7.4.post1 | lldacing | ✅ | ✅ | ✅ | Currently used in install.py, patch required |
| | | | | | | | |

## Test Results Summary

### ✅ PyTorch 2.8.0 + CUDA 12.9 + Flash Attention 2.8.2 (ussoewwin)
- **Status**: ✅ WORKING
- **Tests**: Both `test_kokoro_tts.py` and `test_h2ovl.py` pass
- **Patch**: ❌ **NOT REQUIRED** - Works without patch!
- **Date Tested**: 2025-01-XX
- **Notes**: 
  - This is the latest working combination found
  - PyTorch 2.8.0 with CUDA 12.9 is available
  - Flash Attention 2.8.2 from ussoewwin repository works correctly
  - **IMPORTANT**: Torchvision patch is NOT required for PyTorch 2.8.0 (unlike 2.7.0)
  - Both tests pass with and without the patch

### ❌ Attempted but Failed
- **PyTorch 2.8.0 + CUDA 13.0**: PyTorch 2.8.0 not available for CUDA 13.0
- **PyTorch 2.8.0 + CUDA 12.8 (lldacing)**: Wheel file not found in repository
- **PyTorch 2.9.x**: Not available yet

## Recommendations

### For Latest Version (Recommended)
**Use**: PyTorch 2.8.0 + CUDA 12.9 + Flash Attention 2.8.2 (ussoewwin)
- **Advantages**:
  - Latest stable combination
  - No torchvision patch required
  - Better performance with newer Flash Attention
- **Installation**:
  ```bash
  uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
  # Then install Flash Attention 2.8.2 from ussoewwin repository
  ```

### For Current Stable (Currently Used)
**Use**: PyTorch 2.7.0 + CUDA 12.8 + Flash Attention 2.7.4.post1 (lldacing)
- **Advantages**:
  - Proven stable
  - Well-tested
- **Disadvantages**:
  - Requires torchvision patch
  - Older versions

## Notes
- All tests run on Windows with Python 3.12
- Tests run with and without torchvision patch
- Both test scripts must pass for a version to be considered working
- PyTorch 2.8.0 does NOT require the torchvision patch (unlike 2.7.0)
