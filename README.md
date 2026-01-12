# Visual Novel OCR Tool

A lightweight Python application for extracting text from visual novels and games. Press `Ctrl+Shift+Alt+Z` anywhere to capture the active window, automatically detect text regions, extract text using AI, and copy it to your clipboard. Perfect for reading visual novels in foreign languages or extracting dialogue from games.

## Features

- **One-Key Extraction**: Press `Ctrl+Shift+Alt+Z` to automatically capture, detect, extract, and copy text to clipboard
- **Web UI for Tuning**: Visual interface to fine-tune detection and merge settings with live preview
- **Backend-Authoritative**: All detection/merging logic runs in Python for consistency between preview and extraction
- **Smart Text Detection**: Uses PaddleOCR to detect text regions, then sends only cropped regions to Vision API (reduces costs by 90%+)
- **Auto-Merge Dialogue**: Automatically merges split dialogue lines with configurable tolerances
- **GPU Acceleration**: Automatically uses GPU for text detection if available (PaddleOCR)
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

3. **Install PaddleOCR with GPU support (recommended for better performance)**:   
   
   For GPU support (requires NVIDIA GPU with CUDA):
   ```bash
   # First, check your CUDA version: nvidia-smi
   # PaddlePaddle GPU 2.6.2 supports CUDA 10.2/11.2/11.6/11.7
   # Reference: https://pypi.org/project/paddlepaddle-gpu/
   
   # For CUDA 10.2 (from PyPI - simplest method):
   pip install paddlepaddle-gpu==2.6.2
   
   # For CUDA 11.2/11.6/11.7 (from PaddlePaddle website):
   # Check the official installation guide for your specific CUDA version:
   # https://www.paddlepaddle.org.cn/install/quick
   # The website provides installation commands for different CUDA versions
   
   # Then install PaddleOCR
   pip install paddleocr
   ```
   
   For CPU-only (if no GPU available):
   ```bash
   pip install paddlepaddle==2.6.2
   pip install paddleocr
   ```
   
   **Note**: PyPI only provides CUDA 10.2 builds. For other CUDA versions (11.2/11.6/11.7), 
   you need to install from the PaddlePaddle website. See the [official installation guide](https://www.paddlepaddle.org.cn/install/quick) for details.
   
   **Note**: The application will automatically detect and use GPU if available, otherwise it falls back to CPU. If PaddleOCR is not installed, the app will process the full image instead of detecting text regions.

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

1. **Start the server:**
   ```bash
   uv run python run_server.py
   ```

2. **Open the UI** in your browser: `http://localhost:8000`

3. **Press `Ctrl+Shift+Alt+Z`** while playing a visual novel - that's it! Text is automatically:
   - Captured from the active window
   - Detected using PaddleOCR
   - Merged into complete dialogue lines
   - Extracted using AI
   - **Copied to your clipboard**

4. **Paste anywhere** - the text is ready to use!

The UI automatically updates to show detection boxes and extracted text. You only need to use the UI buttons if you want to tune settings for better detection.

## Usage

### Option 1: Web UI (Recommended for Configuration)

The web UI provides a visual interface to tweak OCR settings with live preview:

```bash
uv run python run_server.py
```

Then open your browser to: `http://localhost:8000`

**Features:**
- **Live Preview**: See detection boxes (red) and merged boxes (blue) update in real-time
- **Visual Sliders**: Adjust all settings with immediate visual feedback
- **Mini Visualizers**: See how min width/height and merge tolerances affect detection
- **Non-destructive Overlays**: Boxes are drawn as HTML overlays, not on the image

**Workflow:**
1. **Quick Start**: Just press `Ctrl+Shift+Alt+Z` while playing a visual novel - text is automatically extracted and copied to clipboard!
2. **Tune Settings** (optional): Use the UI to adjust detection sensitivity and merge tolerances
   - Red boxes show detected text regions
   - Blue boxes show merged dialogue lines
   - Adjust sliders to fine-tune for your game/visual novel
3. **Manual Mode** (optional): Click "New Screenshot" and "Run Detection" if you want to manually test settings
4. The UI automatically shows results from hotkey captures - no need to press buttons unless re-tuning settings

### Option 2: Standalone CLI (No Web UI)

For headless operation without the web UI:

```bash
uv run python -m src.backend.cli
```

This runs the same hotkey functionality but prints results to console instead of updating a web UI. Useful for automation or when you don't need the visual interface.

**Note**: The web UI mode (`run_server.py`) is recommended as it provides visual feedback and allows you to tune settings easily.

## Architecture

- **Backend-Authoritative**: All detection and merging logic runs in Python for consistency
- **Lightweight Frontend**: TypeScript UI only renders results - no duplicate algorithms
- **Debounced Updates**: Slider changes only trigger server requests on release (no spam)
- **Cached Detections**: Detection results are cached per screenshot version for fast preview updates
- **Auto-Sync**: UI automatically reflects hotkey results via Server-Sent Events (SSE)

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
- Make sure you're using the virtual environment: `uv run python run_server.py` or `uv run python run_cli.py`
- Or activate the venv: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac)

### PaddleOCR GPU issues
- Verify CUDA is installed: `nvidia-smi` (should show your GPU)
- Check PaddlePaddle GPU installation: `python -c "import paddle; print(paddle.device.get_device())"`
- If GPU is not detected, the app will automatically use CPU (slower but still works)
- For CUDA version compatibility, check: https://www.paddlepaddle.org.cn/install/quick
