"""
Test script for Kokoro TTS
Tests the TTS model to verify installation and functionality
"""
# CRITICAL: Apply torchvision patch BEFORE importing torch or anything that imports torch
import sys
from pathlib import Path
sys.path.insert(0, 'src')

# Import and apply patch BEFORE any other imports
from backend.core.torchvision_patch import apply_torchvision_patch
apply_torchvision_patch()

import warnings
# Suppress informational warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

print("=" * 60)
print("Kokoro TTS Test Script")
print("=" * 60)
print("\nLoading Kokoro TTS model...")
print("This may take a minute on first run (downloading model ~300MB)...")

try:
    from kokoro import KPipeline
    import soundfile as sf
    import sounddevice as sd
    import numpy as np
    import torch
    
    # Check device
    if torch.cuda.is_available():
        device_type = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device_type = 'cpu'
        print("Using CPU (GPU not available)")
    
    # Initialize pipeline
    print("\nInitializing Kokoro pipeline (lang_code='a' for American English)...")
    print("(This will download the model on first run - may take a few minutes)")
    import warnings
    warnings.filterwarnings("ignore", message=".*No module named pip.*")
    pipeline = KPipeline(lang_code='a')
    print("[OK] Pipeline initialized successfully!")
    
    # Test text
    test_text = '''
    Kokoro is an open-weight TTS model with 82 million parameters. 
    Despite its lightweight architecture, it delivers comparable quality to larger models 
    while being significantly faster and more cost-efficient. 
    With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.
    '''
    
    print("\n" + "-" * 60)
    print("Generating speech from test text...")
    print("-" * 60)
    print(f"Text: {test_text[:100]}...")
    
    # Generate audio
    voice = 'af_heart'  # Default voice
    speed = 1.0
    sample_rate = 24000
    
    print(f"\nVoice: {voice}")
    print(f"Speed: {speed}x")
    print(f"Sample rate: {sample_rate} Hz")
    
    generator = pipeline(test_text, voice=voice, speed=speed, split_pattern=r'\n+')
    
    # Collect all audio segments
    full_audio = []
    segment_count = 0
    
    for i, (gs, ps, audio) in enumerate(generator):
        segment_count += 1
        print(f"  Segment {segment_count}: Generated {len(audio)} samples")
        full_audio.append(audio)
    
    if not full_audio:
        print("\n[WARNING] No audio generated!")
        sys.exit(1)
    
    # Concatenate all segments
    combined = np.concatenate(full_audio)
    duration = len(combined) / sample_rate
    
    print(f"\n[OK] Generated {len(combined)} samples ({duration:.2f} seconds)")
    
    # Save to file
    output_file = Path("test_kokoro_output.wav")
    sf.write(str(output_file), combined, sample_rate)
    print(f"[OK] Saved audio to: {output_file}")
    
    # Play audio
    print("\n" + "-" * 60)
    print("Playing audio...")
    print("-" * 60)
    print("(You should hear the text being spoken now)")
    
    try:
        sd.play(combined, sample_rate)
        sd.wait()  # Wait until playback is finished
        print("\n[OK] Playback completed!")
    except Exception as e:
        print(f"\n[WARNING] Playback error: {e}")
        print("   Audio file was saved, you can play it manually")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print(f"\nOutput file: {output_file.absolute()}")
    
except ImportError as e:
    print(f"\n[ERROR] Import Error: {e}")
    print("\nPlease install required packages:")
    print("  uv pip install kokoro>=0.9.2 soundfile>=0.12.1 sounddevice>=0.4.6")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

