# OCR Accessibility Tool

A Python application that reads text aloud from your screen. Press `Ctrl+Shift+Alt+Z` anywhere to capture the active window, automatically detect text regions, extract text using AI, and read it aloud using Kokoro TTS. Perfect for users who cannot read text on screen - an accessibility tool that makes digital content accessible through voice.

## Features

- **One-Key Reading**: Press `Ctrl+Shift+Alt+Z` to automatically capture, detect, extract, and read text aloud
- **Text-to-Speech**: Always enabled - automatically reads extracted text using Kokoro TTS (82M parameter, Apache-licensed model)
- **Desktop GUI**: Visual interface to fine-tune detection and merge settings with live preview
- **Backend-Authoritative**: All detection/merging logic runs in Python for consistency between preview and extraction
- **Smart Text Detection**: Uses RapidOCR to detect text regions, then processes only cropped regions with the local H2OVL model (faster and more accurate)
- **Auto-Merge Dialogue**: Automatically merges split dialogue lines with configurable tolerances
- **CPU-Based Detection**: RapidOCR runs on CPU via ONNX Runtime (no GPU conflicts with PyTorch)
- **Local AI Model**: Uses H2OVL-Mississippi-0.8B running locally - no API keys or internet required
- **Real-Time Preview**: See detection boxes and merged regions update live as you adjust settings
- **Auto-Refresh UI**: UI automatically reflects results when using hotkey - no manual button presses needed

## Requirements

- Python 3.12+
- `uv` package manager

## Installation

1. **Install `uv`** if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   (Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`)

2. **Install all dependencies** using the installation script:

   The installation script automatically detects your system and installs the appropriate dependencies:
   ```bash
   uv run python install.py
   ```
   
   Or specify GPU or CPU explicitly:
   ```bash
   uv run python install.py --gpu    # Force GPU installation
   uv run python install.py --cpu    # Force CPU-only installation
   ```
   
   **What gets installed:**
   - Base dependencies from `pyproject.toml` (including opencv-python for RapidOCR)
   - PyTorch 2.8.0 (GPU with CUDA 12.9, or CPU version)
   - Flash Attention 2.8.2 (GPU only, Windows prebuilt wheel via Hugging Face Hub)
   - H2OVL-Mississippi dependencies (transformers, accelerate, timm, peft)
   - Kokoro TTS dependencies (kokoro, misaki[en], loguru, soundfile, sounddevice)
   - Spacy model (en-core-web-sm) for Kokoro TTS
   - RapidOCR (CPU via ONNX Runtime, for text detection)
   
   **Requirements for GPU installation**: NVIDIA GPU with CUDA 12.9 support
   
   **Note**: The GPU installation is optional but recommended. The model runs on CPU by default but is significantly faster with a GPU. TTS works on both CPU and GPU.
   
   **Important**: PyTorch 2.8.0 does NOT require the torchvision patch (unlike 2.7.0). If you're using an older version (PyTorch 2.7.0), you can enable the patch by setting the environment variable `TORCHVISION_PATCH=1`.
   
   **Features of the install script:**
   - Auto-detects GPU availability (or use `--gpu`/`--cpu` flags)
   - Uses Hugging Face Hub with XET support for efficient Flash Attention downloads
   - Real-time progress output for all installation steps
   - Automatic verification of all installed components
   - Handles dependency conflicts automatically

## Configuration

The application creates a `config.json` file automatically. You can edit this file or use the GUI to adjust settings.

```json
{
    "max_image_dimension": 1080,
    "preprocessing": {
        "binary_threshold": 0,
        "invert": false,
        "dilation": 0,
        "contrast": 1.0,
        "brightness": 0
    },
    "text_detection": {
        "min_width": 30,
        "min_height": 30,
        "merge_vertical_tolerance": 4,
        "merge_horizontal_tolerance": 50,
        "merge_width_ratio_threshold": 0.3
    },
    "tts": {
        "voice": "af_heart",
        "speed": 1.0
    }
}
```

### Settings Guide:
- **max_image_dimension**: Downscales large images to fit model context (default: 1080)
- **preprocessing**: Image adjustments before detection (useful for difficult backgrounds)
- **text_detection**: Controls how text is detected and merged into lines
- **tts**: Text-to-Speech settings (TTS is always enabled)
  - **voice**: Kokoro voice ID (default: "af_heart")
  - **speed**: Speech speed multiplier (default: 1.0, range: 0.5-2.0)

## Quick Start

### Desktop GUI (Recommended)

The PyQt6 GUI provides instant feedback, real progress bars, and zero latency:

```bash
uv run python run_gui.py
```

**Press `Ctrl+Shift+Alt+Z`** anywhere on your screen - that's it! Text is automatically:
- Captured from the active window
- Detected using RapidOCR
- Merged into complete dialogue lines
- Extracted using AI
- **Read aloud using Kokoro TTS** (always enabled)

The UI automatically updates to show detection boxes and extracted text. You only need to use the UI buttons if you want to tune settings for better detection.

**Features:**
- **Instant Updates**: No server round-trips, everything happens in the same process
- **Real Progress Bars**: See exactly what's happening (detection, merging, OCR)
- **Working Cancellation**: Cancel button actually stops the process immediately
- **Native Performance**: Drawing boxes and updating UI is instant
- **Hotkey Support**: Press `Ctrl+Shift+Alt+Z` anywhere to capture

## Usage

### Desktop GUI (Recommended)

The PyQt6 GUI is the best option for local use - it's faster, more responsive, and has working progress bars:

```bash
uv run python run_gui.py
```

**Why use the GUI:**
- ✅ **Zero latency** - no HTTP requests, everything is instant
- ✅ **Real progress bars** - see exactly what's happening
- ✅ **Working cancellation** - cancel button actually stops processes
- ✅ **Better performance** - native drawing is faster than HTML overlays
- ✅ **Simpler architecture** - no server, no SSE, no state sync issues

**Features:**
- Live preview with bounding boxes drawn directly on the image
- Real-time progress updates during detection and OCR
- Cancel button that actually works
- All settings auto-save to `config.json`
- Hotkey support: Press `Ctrl+Shift+Alt+Z` anywhere

**Workflow:**
1. **Quick Start**: Just press `Ctrl+Shift+Alt+Z` anywhere on your screen - text is automatically extracted and read aloud!
2. **Tune Settings** (optional): Use the UI to adjust detection sensitivity, merge tolerances, and TTS speed
   - Red boxes show detected text regions
   - Blue boxes show merged dialogue lines
   - TTS panel: Adjust speech speed (TTS is always enabled)
   - Adjust sliders to fine-tune for your content
3. **Manual Mode** (optional): Click "New Screenshot" and "Run Detection" if you want to manually test settings
4. The UI automatically shows results from hotkey captures - no need to press buttons unless re-tuning settings

### Standalone CLI (No UI)

For headless operation without the GUI:

```bash
uv run python -m src.backend.cli
```

This runs the same hotkey functionality but prints results to console instead of updating a UI. Text is still automatically read aloud. Useful when you don't need the visual interface.

**Note**: The desktop GUI mode (`run_gui.py`) is recommended as it provides visual feedback and allows you to tune settings easily.

## Architecture

- **Backend-Authoritative**: All detection and merging logic runs in Python for consistency
- **Native GUI**: PyQt6 desktop application with instant updates and real progress tracking
- **Cached Detections**: Detection results are cached per screenshot version for fast preview updates
- **Live Preview**: Size filters are applied after RapidOCR runs, enabling instant preview updates without re-running detection

## Notes

- **Windows**: The `keyboard` library requires administrator privileges for global hotkeys
- **Screenshot**: Captures only the active window (not the full screen)
- **Local Processing**: All OCR processing happens locally - no internet connection or API keys required
- **Accessibility Focus**: This tool is designed to help users who cannot read text on screen - text is always read aloud automatically

## Troubleshooting

### Hotkey not working
- On Windows, run as administrator
- Make sure no other application is using the same hotkey

### Model loading is slow
- The first run downloads the H2OVL model (~1.6GB) and Kokoro TTS model (~300MB) and may take time
- CPU inference is slower than GPU; consider installing CUDA support if you have an NVIDIA card
- TTS model loads on first use - subsequent uses are faster

### TTS not working
- TTS is always enabled - check console for TTS messages - first load may take a minute
- Verify Kokoro is installed: `uv pip list | grep kokoro`
- If you're using PyTorch 2.7.0 and see torchvision errors, enable the patch: `set TORCHVISION_PATCH=1` (Windows) or `export TORCHVISION_PATCH=1` (Linux/Mac) before running

### Import errors
- Make sure you're using the virtual environment: `uv run python run_gui.py` or `uv run python run_cli.py`
- Or activate the venv: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac)

### RapidOCR issues
- RapidOCR runs on CPU via ONNX Runtime - no GPU configuration needed
- RapidOCR is required - install it with: `uv pip install rapidocr-onnxruntime`
- Verify installation: `python -c "from rapidocr_onnxruntime import RapidOCR; print('RapidOCR installed successfully')"`
