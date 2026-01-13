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
from ..state import state
from typing import Dict

# Cached voices dictionary (populated dynamically)
_voices_cache = None
_voices_cache_lock = threading.Lock()

def _format_voice_name(voice_code: str) -> str:
    """
    Format a voice code (e.g., 'af_heart') into a display name (e.g., '🇺🇸 Heart (F)').
    
    Args:
        voice_code: Voice code like 'af_heart', 'am_michael', 'bf_emma', 'bm_george'
    
    Returns:
        Formatted display name with flag emoji and gender indicator
    """
    if not voice_code or '_' not in voice_code:
        return voice_code
    
    prefix, name = voice_code.split('_', 1)
    
    # Determine language and gender from prefix
    if prefix.startswith('af'):
        flag = "🇺🇸"
        gender = "(F)"
        lang_name = "🇺🇸"
    elif prefix.startswith('am'):
        flag = "🇺🇸"
        gender = "(M)"
        lang_name = "🇺🇸"
    elif prefix.startswith('bf'):
        flag = "🇬🇧"
        gender = "(F)"
        lang_name = "🇬🇧"
    elif prefix.startswith('bm'):
        flag = "🇬🇧"
        gender = "(M)"
        lang_name = "🇬🇧"
    else:
        # Unknown prefix, return as-is
        return voice_code
    
    # Capitalize first letter of name
    display_name = name.capitalize()
    
    return f"{lang_name} {display_name} {gender}"

def get_available_voices(repo_id: str = 'hexgrad/Kokoro-82M', use_cache: bool = True) -> dict:
    """
    Dynamically fetch available voices from HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID (default: 'hexgrad/Kokoro-82M')
        use_cache: If True, use cached voices if available
    
    Returns:
        Dictionary mapping display names to voice codes (e.g., {'🇺🇸 Heart (F)': 'af_heart', ...})
    """
    global _voices_cache
    
    # Return cached voices if available and caching is enabled
    if use_cache and _voices_cache is not None:
        return _voices_cache
    
    with _voices_cache_lock:
        # Double-check after acquiring lock
        if use_cache and _voices_cache is not None:
            return _voices_cache
        
        try:
            from huggingface_hub import list_repo_files
            
            print(f"[TTS] Fetching available voices from {repo_id}...")
            files = list_repo_files(repo_id, repo_type='model')
            
            # Filter for voice files (voices/*.pt)
            voice_files = [f for f in files if f.startswith('voices/') and f.endswith('.pt')]
            
            # Extract voice codes (remove 'voices/' prefix and '.pt' suffix)
            voice_codes = [f[7:-3] for f in voice_files]  # 'voices/af_heart.pt' -> 'af_heart'
            
            # Create dictionary mapping display names to voice codes
            voices_dict = {}
            for code in sorted(voice_codes):
                display_name = _format_voice_name(code)
                voices_dict[display_name] = code
            
            print(f"[TTS] Found {len(voices_dict)} available voices")
            
            # Cache the result
            _voices_cache = voices_dict
            
            return voices_dict
            
        except Exception as e:
            print(f"[TTS] ⚠ Failed to fetch voices from repository: {e}")
            print("[TTS] Falling back to default voices...")
            import traceback
            traceback.print_exc()
            
            # Fallback to a minimal set of known working voices
            fallback_voices = {
                "🇺🇸 Heart (F)": "af_heart",
                "🇺🇸 Bella (F)": "af_bella",
                "🇺🇸 Nicole (F)": "af_nicole",
                "🇺🇸 Sarah (F)": "af_sarah",
                "🇺🇸 Michael (M)": "am_michael",
                "🇺🇸 Adam (M)": "am_adam",
                "🇬🇧 Emma (F)": "bf_emma",
                "🇬🇧 George (M)": "bm_george",
            }
            
            # Cache fallback to avoid repeated failures
            if _voices_cache is None:
                _voices_cache = fallback_voices
            
            return fallback_voices

def clear_voices_cache():
    """Clear the cached voices list to force a refresh on next fetch"""
    global _voices_cache
    with _voices_cache_lock:
        _voices_cache = None

# Global accessor for VOICES (for backward compatibility)
def get_voices() -> dict:
    """Get available voices dictionary (dynamically fetched from HuggingFace)"""
    return get_available_voices()

# For backward compatibility, VOICES is now a function that returns the dict
# This allows existing code like "from src.backend.core.tts import VOICES" to work
# but VOICES is now callable: VOICES() or use get_voices()
def VOICES() -> dict:
    """Get available voices (backward compatibility wrapper)"""
    return get_available_voices()

# Global Queue for TTS tasks
_tts_queue = queue.Queue()
_tts_thread = None
_stop_event = threading.Event()

# Pipeline singleton (supports multiple lang codes)
_tts_pipelines = {}  # Dict of lang_code -> pipeline
_tts_lock = threading.Lock()

def get_pipeline(lang_code='a'):
    """
    Lazy load the Kokoro pipeline for a specific language code.
    
    Args:
        lang_code: 'a' for American English, 'b' for British English
    
    Returns:
        KPipeline instance or None if failed
    """
    global _tts_pipelines
    with _tts_lock:
        if lang_code not in _tts_pipelines:
            try:
                print(f"[TTS] Loading Kokoro model (lang_code='{lang_code}')...")
                from kokoro import KPipeline
                # 'a' = American English. 'b' = British English.
                # triggers download on first run (~300MB)
                _tts_pipelines[lang_code] = KPipeline(lang_code=lang_code) 
                print(f"[TTS] Model loaded (lang_code='{lang_code}').")
            except Exception as e:
                print(f"[TTS] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                return None
    return _tts_pipelines.get(lang_code)

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
    Stores audio and phonemes in state for replay functionality.
    """
    while not _stop_event.is_set():
        try:
            # Wait for text (timeout allows checking stop_event occasionally)
            item = _tts_queue.get(timeout=1.0)
            # Support both old format (text, voice, speed) and new format (text, voice, speed, volume)
            if len(item) == 4:
                text, voice, speed, volume = item
            else:
                text, voice, speed = item
                volume = 1.0  # Default volume
            
            # Determine lang_code from voice prefix (af_/am_ = 'a', bf_/bm_ = 'b')
            lang_code = 'b' if voice.startswith('b') else 'a'
            pipeline = get_pipeline(lang_code)
            
            if pipeline and text:
                try:
                    # Generate audio with phoneme information
                    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
                    
                    full_audio = []
                    all_phonemes = []
                    sample_rate = 24000
                    
                    # Process each segment (gs = graphemes, ps = phonemes, audio = audio data)
                    for gs, ps, audio in generator:
                        if _stop_event.is_set():
                            break
                        # Store raw audio (without volume) for replay
                        full_audio.append(audio)
                        if ps:  # Collect phonemes
                            all_phonemes.append(ps)
                        # Apply volume and play immediately for streaming feel
                        audio_with_volume = audio * volume
                        sd.play(audio_with_volume, sample_rate)
                        sd.wait()
                    
                    if full_audio:
                        # Store raw audio in state for replay (volume will be applied on replay)
                        combined = np.concatenate(full_audio)
                        state.last_audio_data = combined
                        state.last_phonemes = " ".join(all_phonemes) if all_phonemes else ""
                        
                except Exception as e:
                    print(f"[TTS] Generation/Playback error: {e}")
                    import traceback
                    traceback.print_exc()
            
            _tts_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS] Worker error: {e}")
            import traceback
            traceback.print_exc()

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
    config = load_config()
    tts_config = config.get("tts", {})
    voice = tts_config.get("voice", "af_heart")
    
    # Determine lang_code from voice prefix (af_/am_ = 'a', bf_/bm_ = 'b')
    lang_code = 'b' if voice.startswith('b') else 'a'
    pipeline = get_pipeline(lang_code)
    
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
    volume = tts_config.get("volume", 1.0)  # Default to 100% volume
    
    # Add to queue with volume
    print(f"[TTS] Queued: {text[:50]}...")
    _tts_queue.put((text, voice, speed, volume))

