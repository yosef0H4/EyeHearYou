Here is a detailed, readable summary of the **OCR Accessibility Tool**.

### 🎯 Core Purpose
A Python application designed for accessibility that captures your screen (or specific active window), uses local AI to extract text, and automatically reads it aloud. It is optimized for Visual Novels and games but works on any screen content.

### ✨ Key Features
*   **One-Key Reading:** Press `Ctrl+Shift+Alt+Z` to capture, detect, extract, and read text instantly.
*   **Local & Offline:** Uses **H2OVL-Mississippi-0.8B** for high-accuracy OCR and **Kokoro TTS** for natural voice generation. No internet required after setup.
*   **Smart & Manual Selection:**
    *   **Smart:** Uses RapidOCR to auto-detect text regions.
    *   **Manual:** Draw permanent boxes (resolution independent) for fixed dialogue areas.
    *   **Refinement:** "Photoshop-style" Add/Subtract tools to fine-tune detection areas.
*   **Configuration Profiles:** Save specific settings (detection sensitivity, voice, boxes) for different games. The "Default" profile is locked as a safety net.
*   **Multi-Language UI:** Fully translated interface in **English** and **Arabic** (with RTL support).
*   **Advanced Text Merging:** Intelligently merges split sentences and paragraphs for smooth reading.

### ⚠️ Hardware Requirements
*   **Recommended:** NVIDIA GPU with CUDA (minimum 6GB VRAM). Performance is fast (1-3 seconds).
*   **Not Recommended:** CPU-only mode. It is untested and will be very slow (10-30+ seconds per capture).

### 🚀 Installation & Usage
1.  **Install:** Requires Python 3.12+ and `uv`.
    *   Run `uv run python install.py` (Auto-detects GPU/CPU).
2.  **Run:** `uv run python run_gui.py`
3.  **Hotkeys:**
    *   `Ctrl+Shift+Alt+Z`: Extract & Read.
    *   `Ctrl+Shift+Alt+X`: Detect Only (Preview).

### ⚙️ Configuration Settings
*   **Preprocessing:** Adjust contrast, brightness, or binarize text (useful for transparent text boxes).
*   **Text Sorting:** Supports Horizontal (LTR/RTL) and Vertical (LTR/RTL) reading orders (e.g., for Manga).
*   **TTS:** Adjust speed (0.5x - 2.0x) and choose from available Kokoro voices.

---

### 🙌 Acknowledgements (Shouts)
This project would not be possible without the incredible work of the following open-source projects, models, and communities:

*   **[RapidOCR](https://github.com/RapidAI/RapidOCR)** - Fast and accurate text detection using ONNX Runtime. This project uses RapidOCR for initial text region detection, which can run on both CPU and GPU (with onnxruntime-gpu) for ~27% faster performance.
*   **[H2OVL-Mississippi-0.8B](https://huggingface.co/h2oai/h2ovl-mississippi-800m)** - The vision-language model from H2O.ai that powers the text extraction. This 800M parameter model provides high-accuracy OCR locally without requiring internet connectivity or API keys.
*   **[Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M)** - The 82M parameter text-to-speech model by hexgrad that reads extracted text aloud. Kokoro provides high-quality, natural-sounding speech with Apache-licensed weights, making it perfect for offline accessibility applications.
*   **[Flash Attention 2 for Windows](https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows)** - Prebuilt Windows wheels for Flash Attention 2.8.2 that enable efficient GPU acceleration for transformer models. The prebuilt wheels from ussoewwin make it possible to use Flash Attention on Windows without complex compilation.
*   **[comic-translate](https://github.com/ogkalu2/comic-translate)** by **[ogkalu2](https://github.com/ogkalu2)** - This project was heavily inspired by comic-translate and borrows several key algorithms including text region filtering, text box merging logic, reading order sorting, and contained box filtering.