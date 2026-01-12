# Screenshot OCR Application

A Python application that captures screenshots on hotkey press (Ctrl+Shift+Alt+Z) and extracts text using OpenAI Vision API. Compatible with LM Studio and any OpenAI-compatible API.

## Features

- **Global Hotkey**: Press `Ctrl+Shift+Alt+Z` to capture active window and extract text
- **OpenAI Vision API**: Uses Vision API for accurate text extraction
- **Cost-Effective**: Detects text regions first, then sends only cropped regions to API (reduces costs significantly)
- **GPU Support**: Automatically uses GPU for text detection if available (PaddleOCR)
- **Configurable**: Customize API URL, key, and model via `config.json`
- **LM Studio Compatible**: Works out of the box with LM Studio

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
1. Click "New Screenshot" to capture the active window
2. Click "Run Detection" to detect text regions (red boxes appear)
3. Adjust merge settings (vertical/horizontal tolerance, width ratio) - blue boxes update live
4. Click "EXTRACT TEXT" to get the final OCR result

### Option 2: Command Line (Hotkey Mode)

Run the standalone CLI:

```bash
uv run python -m src.backend.cli
```

Or use the convenience script (if you create one):

```bash
uv run python run_cli.py
```

The application will:
1. Load configuration from `config.json`
2. Register the global hotkey `Ctrl+Shift+Alt+Z`
3. Wait for the hotkey press

When you press `Ctrl+Shift+Alt+Z`:
1. Captures the active window
2. Detects text regions using PaddleOCR (if installed)
3. Sends only the detected text regions to the Vision API (much cheaper!)
4. Extracts and prints text to the console

Press `Ctrl+C` to exit.

## Notes

- **Windows**: The `keyboard` library requires administrator privileges for global hotkeys
- **Screenshot**: Captures only the active window (not the full screen)
- **API Compatibility**: Works with any OpenAI-compatible API endpoint

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
