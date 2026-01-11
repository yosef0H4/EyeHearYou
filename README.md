# Screenshot OCR Application

A Python application that captures screenshots on hotkey press (Ctrl+Shift+Alt+Z) and extracts text using OpenAI Vision API. Compatible with LM Studio and any OpenAI-compatible API.

## Features

- **Global Hotkey**: Press `Ctrl+Shift+Alt+Z` to capture active window and extract text
- **OpenAI Vision API**: Uses Vision API for accurate text extraction
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

2. Install dependencies (already done if you used `uv init`):
   ```bash
   uv sync
   ```

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

Run the application:

```bash
uv run python main.py
```

The application will:
1. Load configuration from `config.json`
2. Register the global hotkey `Ctrl+Shift+Alt+Z`
3. Wait for the hotkey press

When you press `Ctrl+Shift+Alt+Z`:
1. Captures the active window
2. Sends the screenshot to the Vision API
3. Extracts and prints text to the console

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
- Make sure you're using the virtual environment: `uv run python main.py`
- Or activate the venv: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac)
