"""Text-to-Speech functionality using Kokoro with Queueing"""
# Torchvision patch is only needed for PyTorch 2.7.0
# PyTorch 2.8.0+ does NOT require the patch
# To enable patch for older versions, set environment variable: TORCHVISION_PATCH=1
import os
if os.getenv("TORCHVISION_PATCH", "0") == "1":
    from .torchvision_patch import apply_torchvision_patch
    apply_torchvision_patch()

import threading
import queue
import sounddevice as sd
import numpy as np
import torch
from .config import load_config

# Global Queue for TTS tasks
_tts_queue = queue.Queue()
_tts_thread = None
_stop_event = threading.Event()

# Pipeline singleton
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

def _tts_worker():
    """
    Dedicated worker thread that processes text from the queue sequentially.
    This ensures sentences are read in order and don't overlap.
    """
    while not _stop_event.is_set():
        try:
            # Wait for text (timeout allows checking stop_event occasionally)
            item = _tts_queue.get(timeout=1.0)
            text, voice, speed = item
            
            pipeline = get_pipeline()
            if pipeline and text:
                try:
                    # Generate audio
                    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
                    
                    full_audio = []
                    sample_rate = 24000
                    
                    for _, _, audio in generator:
                        full_audio.append(audio)
                    
                    if full_audio:
                        combined = np.concatenate(full_audio)
                        # Blocking playback ensures we don't start next sentence until this one finishes
                        sd.play(combined, sample_rate)
                        sd.wait()
                        
                except Exception as e:
                    print(f"[TTS] Generation/Playback error: {e}")
            
            _tts_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS] Worker error: {e}")

def start_tts_engine():
    """Ensure the background TTS thread is running"""
    global _tts_thread
    if _tts_thread is None or not _tts_thread.is_alive():
        _stop_event.clear()
        _tts_thread = threading.Thread(target=_tts_worker, daemon=True, name="TTS_Worker")
        _tts_thread.start()

def stop_tts_engine():
    """Clear queue and stop audio"""
    # Clear pending items
    while not _tts_queue.empty():
        try:
            _tts_queue.get_nowait()
        except queue.Empty:
            break
            
    # Stop current audio
    try:
        sd.stop()
    except:
        pass

def preload_tts(test=True):
    """
    Preload the Kokoro TTS pipeline at startup.
    
    Args:
        test: If True, run a simple test to verify TTS works
        
    Returns:
        True if TTS pipeline loaded (and tested) successfully, False otherwise
    """
    print("[TTS] Preloading Kokoro TTS model...")
    pipeline = get_pipeline()
    
    if pipeline is None:
        print("[TTS] ⚠ Failed to load TTS pipeline")
        return False
    
    # Start the TTS worker thread
    start_tts_engine()
    
    if test:
        print("[TTS] Running startup test...")
        try:
            # Run a simple test with a short phrase
            test_text = "TTS ready"
            config = load_config()
            tts_config = config.get("tts", {})
            voice = tts_config.get("voice", "af_heart")
            speed = tts_config.get("speed", 1.0)
            
            # Generate audio (but don't play it)
            generator = pipeline(test_text, voice=voice, speed=speed, split_pattern=r'\n+')
            audio_segments = list(generator)
            
            if audio_segments:
                print("[TTS] ✓ TTS pipeline loaded and tested successfully!")
                return True
            else:
                print("[TTS] ⚠ TTS pipeline loaded but test returned no audio")
                return True  # Still return True, pipeline is loaded
        except Exception as e:
            print(f"[TTS] ⚠ TTS pipeline loaded but test failed: {e}")
            return True  # Still return True, pipeline is loaded
    else:
        print("[TTS] ✓ TTS pipeline loaded successfully!")
        return True

def speak_text(text: str, clear_queue: bool = False):
    """
    Queue text to be spoken.
    
    Args:
        text: The text to speak
        clear_queue: If True, stops current audio and clears pending messages (good for fresh captures)
    """
    if not text or not text.strip():
        return

    # Ensure engine is running
    start_tts_engine()

    if clear_queue:
        stop_tts_engine()

    config = load_config()
    tts_config = config.get("tts", {})
    voice = tts_config.get("voice", "af_heart")
    speed = tts_config.get("speed", 1.0)
    
    # Add to queue
    print(f"[TTS] Queued: {text[:50]}...")
    _tts_queue.put((text, voice, speed))

