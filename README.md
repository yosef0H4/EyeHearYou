# OCR Accessibility Tool

A Python application that reads text aloud from your screen. Press `Ctrl+Shift+Alt+Z` anywhere to capture the active window, automatically detect text regions, extract text using AI, and read it aloud using Kokoro TTS. Perfect for users who cannot read text on screen - an accessibility tool that makes digital content accessible through voice.

## Features

- **One-Key Reading**: Press `Ctrl+Shift+Alt+Z` to automatically capture, detect, extract, and read text aloud.
- **Smart & Manual Selection**:
  - **Smart Detection**: Uses RapidOCR to automatically find text on screen (toggleable).
  - **Manual Boxes**: Draw permanent boxes over fixed areas (like dialogue boxes in Visual Novels). These persist across screenshots, adapt to window resizing, and are saved to config.
  - **Area Selection**: "Photoshop-style" Add/Subtract selection tools to precisely define where to look (or what to ignore). Uses normalized coordinates for resolution independence.
- **Text-to-Speech**: Always enabled - automatically reads extracted text using Kokoro TTS (82M parameter, Apache-licensed model).
- **Local AI Model**: Uses H2OVL-Mississippi-0.8B running locally for high-accuracy text extraction (no internet required).
- **Auto-Merge Dialogue**: Intelligent merging of split sentences and paragraphs using adaptive logic that works across all screen sizes.
- **Real-Time Preview**: Native PyQt6 GUI with instant visual feedback for tuning detection settings.
- **Resolution Independent**: Manual boxes and selection areas use normalized coordinates (0-1), so they work correctly even if you resize the game window or change resolution.
- **Contained Box Filtering**: Automatically removes smaller boxes that are mostly inside larger manual boxes (keeps the container, discards the inner ones).
- **Loading Screen**: Shows progress bar during model warm-up on startup.

## Requirements

- Python 3.12+
- `uv` package manager (recommended) or standard pip.
- **Hardware**: NVIDIA GPU (6GB+ VRAM) recommended for speed, but runs on CPU (slower).

## Installation

1. **Install `uv`** (Fast Python package installer):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   (Windows PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`)

2. **Install Dependencies**:
   Run the automated install script to detect your hardware and install the correct versions of PyTorch, Flash Attention, and models:

   ```bash
   uv run python install.py
   ```
   
   *Options:*
   - `uv run python install.py --gpu` (Force GPU installation)
   - `uv run python install.py --cpu` (Force CPU-only installation)

   **What gets installed:**
   - Base dependencies from `pyproject.toml` (including opencv-python for RapidOCR)
   - PyTorch 2.8.0 (GPU with CUDA 12.9, or CPU version)
   - Flash Attention 2.8.2 (GPU only, Windows prebuilt wheel via Hugging Face Hub)
   - H2OVL-Mississippi dependencies (transformers, accelerate, timm, peft)
   - Kokoro TTS dependencies (kokoro, misaki[en], loguru, soundfile, sounddevice)
   - Spacy model (en-core-web-sm) for Kokoro TTS
   - RapidOCR (CPU via ONNX Runtime, for text detection)

## Quick Start

1. **Run the GUI**:
   ```bash
   uv run python run_gui.py
   ```

2. **Wait for Models to Load**: A loading screen will show progress as models warm up (first run may take a minute to download models).

3. **Capture**:
   - Switch to your game or application.
   - Press **`Ctrl+Shift+Alt+Z`** (Extract + Read) or **`Ctrl+Shift+Alt+X`** (Detect Only).
   - The tool captures the window, extracts text, and starts reading immediately.

4. **Tune (Optional)**:
   - Switch back to the GUI to see the preview.
   - Use the **Selection Tools** to refine exactly what text is read.

## Selection Tools Guide

The GUI provides powerful tools to control exactly what text is processed.

### Toolbar Modes
- **✋ View**: Pan and zoom the image preview.
- **➕ Add Area**: Draw a rectangle to *include* this area for Smart Detection. (By default, the whole image is included).
- **➖ Remove Area**: Draw a rectangle to *exclude* text in this area (e.g., to ignore a chat window, UI elements, or subtitles).
- **📦 Manual Box**: Draw a **permanent** box.
  - **Usage**: Perfect for Visual Novels where the text box is always in the same place.
  - **Behavior**: These boxes **persist** even if you close the app. They use normalized coordinates (0-1), so they stay correct even if you resize the game window or change resolution.
  - **Editing**: Hover over a manual box and click the red **×** to delete it.
  - **Clear All**: Use the "Clear Manual" button to remove all manual boxes at once.

### Actions
- **Select All**: Reset to default state (everything selected).
- **Deselect All**: Clear all selections (nothing selected).
- **Clear Manual**: Remove all manually drawn boxes.

### Workflow Examples

**Scenario A: Standard Visual Novel (Fixed Text Box)**
1. Uncheck "Use Smart Detection (RapidOCR)".
2. Select **📦 Manual Box**.
3. Draw a box over the dialogue area.
4. Done! Now every time you press the hotkey, it only reads that specific area. The box persists across sessions and adapts to window resizing.

**Scenario B: RPG with Dynamic Text bubbles**
1. Check "Use Smart Detection (RapidOCR)".
2. If the tool is reading UI numbers (HP/MP) you don't want:
   - Select **➖ Remove Area**.
   - Draw boxes over the HP/MP bars.
3. Now the tool will detect text *everywhere* except those specific spots.

**Scenario C: Mixed Mode (Smart Detection + Manual Safety Net)**
1. Check "Use Smart Detection (RapidOCR)".
2. Draw a **📦 Manual Box** over the dialogue area as a backup.
3. If RapidOCR misses the dialogue, the manual box will catch it.
4. The contained box filter automatically removes RapidOCR detections that are inside your manual box (keeps the larger manual box).

## Configuration Settings

Settings are saved to `config.json`. Key parameters:

- **max_image_dimension**: Downscales huge screenshots to save memory (Default: 1080).
- **preprocessing**: Adjusts image contrast/brightness before OCR (useful for transparent text boxes).
  - *binary_threshold*: Binarization threshold (0-255, 0=disabled)
  - *invert*: Invert colors (useful for dark text on light backgrounds)
  - *dilation*: Thicken text (0-5, useful for thin fonts)
  - *contrast*: Contrast multiplier (0.5-3.0)
  - *brightness*: Brightness adjustment (-100 to 100)
- **text_detection**:
  - *min_height_ratio*: Filters out tiny text as fraction of screen height (default: 0.031 = 3.1%).
  - *min_width_ratio*: Minimum width as fraction of screen width (default: 0.0 = disabled).
  - *median_height_fraction*: Discard boxes smaller than this fraction of median text size (default: 1.0 = 100%, less aggressive noise filtering).
  - *merge_vertical_ratio*: Vertical gap as multiplier of text height for merging lines (default: 0.07 = tight vertical merging).
  - *merge_horizontal_ratio*: Horizontal gap as multiplier of text height for merging words (default: 0.37 = tight horizontal merging).
  - *merge_width_ratio_threshold*: Minimum horizontal overlap ratio for vertical merging (default: 0.75).
- **text_sorting**:
  - *direction*: Reading order ("horizontal_ltr", "horizontal_rtl", "vertical_ltr", "vertical_rtl").
  - *group_tolerance*: Line grouping tolerance multiplier (default: 0.5).
- **tts**:
  - *speed*: Speech speed multiplier (0.5-2.0, default: 1.0).
  - *voice*: Kokoro voice ID (default: "af_heart").
- **manual_boxes**: Automatically saved list of manual boxes in normalized coordinates (persists across sessions).

## Hotkeys

- **`Ctrl+Shift+Alt+Z`**: Capture screenshot, detect text, extract text, and read aloud (full extraction).
- **`Ctrl+Shift+Alt+X`**: Capture screenshot and detect text only (preview mode, no extraction).

## Testing

Test scripts are available to verify installation:

- **Test OCR Model**: `python test_ocr_model.py`
  - Tests H2OVL model on `test.png`
  - Shows CUDA version, PyTorch version, and Flash Attention status
  - Displays extracted text from test image

- **Test TTS**: `python test_kokoro_tts.py`
  - Tests Kokoro TTS installation
  - Generates and plays test audio

## Troubleshooting

- **Hotkey not working?** Run the terminal/command prompt as Administrator (Windows restriction).
- **Slow performance?** Ensure you installed with `--gpu` and have CUDA installed. CPU mode works but takes 2-5 seconds per capture.
- **Manual Boxes disappeared?** They are saved in `config.json` under `manual_boxes`. If you deleted the config file, they are reset.
- **Text box selected but not reading?** Check the "Preprocessing" tab. Try increasing "Contrast" or using "Binary Threshold" if the text blends into the background.
- **Loading screen stuck?** First run downloads models (~1.6GB H2OVL + ~300MB Kokoro). Check your internet connection and disk space.
- **Flash Attention warnings?** These are informational - the model will use SDPA fallback if Flash Attention isn't available. GPU performance may be slower without Flash Attention.

## Architecture

- **Frontend**: PyQt6 (Native Desktop GUI) for zero-latency interactions.
- **Backend**: 
  - **RapidOCR (ONNX)**: CPU-optimized initial text detection.
  - **H2OVL-Mississippi (PyTorch)**: Vision-Language Model for accurate text recognition.
  - **Kokoro (ONNX/PyTorch)**: High-quality offline Text-to-Speech.
- **State Management**: Centralized state with normalized coordinates for resolution-independent selections.
- **Worker Threads**: OCR and TTS run in background threads to keep UI responsive.
- **Caching**: Detection results are cached per screenshot version for fast preview updates.

## Notes

- **Windows**: The `keyboard` library requires administrator privileges for global hotkeys.
- **Screenshot**: Captures only the active window (not the full screen).
- **Local Processing**: All OCR processing happens locally - no internet connection or API keys required (except for initial model downloads).
- **Accessibility Focus**: This tool is designed to help users who cannot read text on screen - text is always read aloud automatically.
- **Normalized Coordinates**: Manual boxes and selections use 0-1 coordinates, making them work across different resolutions and aspect ratios.
