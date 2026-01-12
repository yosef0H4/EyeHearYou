"""Text-to-Speech functionality using Kokoro"""
# Torchvision patch is only needed for PyTorch 2.7.0
# PyTorch 2.8.0+ does NOT require the patch
# To enable patch for older versions, set environment variable: TORCHVISION_PATCH=1
import os
if os.getenv("TORCHVISION_PATCH", "0") == "1":
    from .torchvision_patch import apply_torchvision_patch
    apply_torchvision_patch()

import threading
import sounddevice as sd
import numpy as np
import torch
from .config import load_config
from .task_manager import task_manager

_tts_pipeline = None
_tts_lock = threading.Lock()

def get_pipeline():
    """Lazy load the Kokoro pipeline"""
    global _tts_pipeline
    with _tts_lock:
        if _tts_pipeline is None:
            try:
                print("[TTS] Loading Kokoro model...")
                from kokoro import KPipeline
                # 'a' = American English. Use 'b' for British.
                # triggers download on first run (~300MB)
                _tts_pipeline = KPipeline(lang_code='a') 
                print("[TTS] Model loaded.")
            except Exception as e:
                print(f"[TTS] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                return None
    return _tts_pipeline

def play_audio(audio, sample_rate):
    """Play audio data using sounddevice"""
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"[TTS] Playback error: {e}")

def _speak_thread(text: str, voice: str = 'af_heart', speed: float = 1.0):
    """Background thread to generate and play audio"""
    try:
        pipeline = get_pipeline()
        if not pipeline:
            return

        # Split text into manageable chunks if needed, but Kokoro handles sentences well.
        # generate() returns an iterator
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
        
        # Concatenate all audio segments to play smoothly
        # (Or play them one by one. Concatenation is smoother for short texts)
        full_audio = []
        sample_rate = 24000
        
        for _, _, audio in generator:
            full_audio.append(audio)
            
        if full_audio:
            # Concatenate numpy arrays
            combined = np.concatenate(full_audio)
            play_audio(combined, sample_rate)
            
    except Exception as e:
        print(f"[TTS] Error generating audio: {e}")
        import traceback
        traceback.print_exc()

def speak_text(text: str):
    """
    Speak the provided text if TTS is enabled in config.
    Runs in a separate thread to not block the main application.
    """
    config = load_config()
    tts_config = config.get("tts", {})
    
    if not tts_config.get("enabled", False):
        print("[TTS] TTS is disabled in config. Enable it in the GUI settings or add 'tts': {'enabled': true} to config.json")
        return

    if not text or not text.strip():
        print("[TTS] No text to speak (empty or whitespace only)")
        return

    voice = tts_config.get("voice", "af_heart")
    speed = tts_config.get("speed", 1.0)
    
    print(f"[TTS] Speaking text ({len(text)} chars) with voice '{voice}' at speed {speed}x...")
    
    # Run in background thread
    t = threading.Thread(target=_speak_thread, args=(text, voice, speed), daemon=True)
    t.start()

