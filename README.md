# Visual Novel OCR Tool

A lightweight Python application for extracting text from visual novels and games. Press `Ctrl+Shift+Alt+Z` anywhere to capture the active window, automatically detect text regions, extract text using AI, and copy it to your clipboard. Perfect for reading visual novels in foreign languages or extracting dialogue from games.

## Features

- **One-Key Extraction**: Press `Ctrl+Shift+Alt+Z` to automatically capture, detect, extract, and copy text to clipboard
- **Desktop GUI**: Visual interface to fine-tune detection and merge settings with live preview
- **Backend-Authoritative**: All detection/merging logic runs in Python for consistency between preview and extraction
- **Smart Text Detection**: Uses RapidOCR to detect text regions, then sends only cropped regions to Vision API (reduces costs by 90%+)
- **Auto-Merge Dialogue**: Automatically merges split dialogue lines with configurable tolerances
- **CPU-Based Detection**: RapidOCR runs on CPU via ONNX Runtime (no GPU conflicts with PyTorch)
- **OpenAI-Compatible**: Works with LM Studio, OpenAI API, or any OpenAI-compatible endpoint
- **Real-Time Preview**: See detection boxes and merged regions update live as you adjust settings
- **Auto-Refresh UI**: UI automatically reflects results when using hotkey - no manual button presses needed

## Requirements

- Python 3.12+
- `uv` package manager

## Installation

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install base dependencies:
   ```bash
   uv sync
   ```

3. **Install RapidOCR (for text detection)**:
   ```bash
   uv pip install rapidocr-onnxruntime
   ```
   
   **Note**: RapidOCR runs on CPU via ONNX Runtime and doesn't require GPU or CUDA. It's lightweight and doesn't conflict with PyTorch dependencies. If RapidOCR is not installed, the app will process the full image instead of detecting text regions.

4. **Optional: Install GPU support for H2OVL-Mississippi OCR model**:
   
   If you want to use the H2OVL-Mississippi-0.8B model for OCR (instead of API-based OCR), run:
   ```bash
   install-gpu.bat
   ```
   
   This installs:
   - PyTorch 2.7.0 with CUDA 12.8
   - Flash Attention 2.7.4 (Windows prebuilt wheel)
   - H2OVL-Mississippi dependencies (transformers, accelerate, timm, peft)
   
   **Requirements**: NVIDIA GPU with CUDA 12.8 support
   
   **Note**: The GPU installation is optional. The default OCR uses API-based extraction (OpenAI, LM Studio, etc.) which doesn't require GPU.

## Configuration

Edit `config.json` to set your API settings:

```json
{
    "api_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model": "gpt-4-vision-preview"
}
```

### For LM Studio:
- **api_url**: Usually `http://localhost:1234/v1` (default LM Studio port)
- **api_key**: Can be any string (e.g., `lm-studio`)
- **model**: The model name you're using in LM Studio (must support vision)

### For OpenAI:
- **api_url**: `https://api.openai.com/v1`
- **api_key**: Your OpenAI API key
- **model**: `gpt-4-vision-preview` or `gpt-4o`

### For Other OpenAI-Compatible APIs:
- Set the **api_url** to your API endpoint
- Set the **api_key** as required by your provider
- Set the **model** to a vision-capable model name

## Quick Start

### Desktop GUI (Recommended)

The PyQt6 GUI provides instant feedback, real progress bars, and zero latency:

```bash
uv run python run_gui.py
```

**Press `Ctrl+Shift+Alt+Z`** while playing a visual novel - that's it! Text is automatically:
- Captured from the active window
- Detected using RapidOCR
- Merged into complete dialogue lines
- Extracted using AI
- **Copied to your clipboard**

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
1. **Quick Start**: Just press `Ctrl+Shift+Alt+Z` while playing a visual novel - text is automatically extracted and copied to clipboard!
2. **Tune Settings** (optional): Use the UI to adjust detection sensitivity and merge tolerances
   - Red boxes show detected text regions
   - Blue boxes show merged dialogue lines
   - Adjust sliders to fine-tune for your game/visual novel
3. **Manual Mode** (optional): Click "New Screenshot" and "Run Detection" if you want to manually test settings
4. The UI automatically shows results from hotkey captures - no need to press buttons unless re-tuning settings

### Standalone CLI (No UI)

For headless operation without the web UI:

```bash
uv run python -m src.backend.cli
```

This runs the same hotkey functionality but prints results to console instead of updating a UI. Useful for automation or when you don't need the visual interface.

**Note**: The desktop GUI mode (`run_gui.py`) is recommended as it provides visual feedback and allows you to tune settings easily.

## Architecture

- **Backend-Authoritative**: All detection and merging logic runs in Python for consistency
- **Native GUI**: PyQt6 desktop application with instant updates and real progress tracking
- **Cached Detections**: Detection results are cached per screenshot version for fast preview updates
- **Live Preview**: Size filters are applied after RapidOCR runs, enabling instant preview updates without re-running detection

## Notes

- **Windows**: The `keyboard` library requires administrator privileges for global hotkeys
- **Screenshot**: Captures only the active window (not the full screen)
- **API Compatibility**: Works with any OpenAI-compatible API endpoint
- **Clipboard**: Requires `pyperclip` package (included in dependencies) or falls back to system-specific methods

## Troubleshooting

### Hotkey not working
- On Windows, run as administrator
- Make sure no other application is using the same hotkey

### API errors
- Verify your `config.json` settings
- Check that your API endpoint is running (for LM Studio)
- Ensure the model supports vision capabilities

### Import errors
- Make sure you're using the virtual environment: `uv run python run_gui.py` or `uv run python run_cli.py`
- Or activate the venv: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac)

### RapidOCR issues
- RapidOCR runs on CPU via ONNX Runtime - no GPU configuration needed
- If RapidOCR is not installed, the app will process the full image instead of detecting text regions
- Verify installation: `python -c "from rapidocr_onnxruntime import RapidOCR; print('RapidOCR installed successfully')"`
