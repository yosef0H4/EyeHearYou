"""
Test script for H2OVL-Mississippi-0.8B OCR model
Tests the model against test.png to verify installation and functionality

Usage:
    python test_h2ovl.py [--no-patch]
    
    --no-patch: Skip applying the torchvision compatibility patch
"""
import argparse
import warnings
import sys
from pathlib import Path
import importlib.util

# Parse arguments
parser = argparse.ArgumentParser(description="Test H2OVL OCR model installation")
parser.add_argument("--no-patch", action="store_true", help="Skip applying torchvision compatibility patch")
args = parser.parse_args()

# CRITICAL: Apply torchvision patch BEFORE importing torch or anything that imports torch
# (unless --no-patch is specified)
if not args.no_patch:
    # Import patch module directly without triggering __init__.py
    spec = importlib.util.spec_from_file_location("torchvision_patch", "src/backend/core/torchvision_patch.py")
    torchvision_patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(torchvision_patch)
    torchvision_patch.apply_torchvision_patch()
    print("[INFO] Torchvision patch applied")
else:
    print("[INFO] Torchvision patch SKIPPED (--no-patch flag)")

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from PIL import Image

# Suppress informational warnings (optional - remove if you want to see all warnings)
# Note: Some warnings appear during model loading (before this code runs) and cannot be suppressed here
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*Flash Attention 2 only supports.*")
warnings.filterwarnings("ignore", message=".*Flash Attention is not available.*")
warnings.filterwarnings("ignore", message=".*FlashAttention is not installed.*")
# Autocast is used during inference to optimize performance and address Flash Attention 2 recommendations

# Set up the model and tokenizer
model_path = 'h2oai/h2ovl-mississippi-800m'

print("=" * 60)
print("H2OVL-Mississippi-0.8B Test Script")
print("=" * 60)
print(f"\nLoading model from: {model_path}")
print("This may take a minute on first run (downloading model)...")

# Check if Flash Attention is available
use_flash_attention = False
try:
    import flash_attn
    use_flash_attention = True
    print("\n[OK] Flash Attention 2 is available")
except (ImportError, OSError) as e:
    print(f"\n[WARNING] Flash Attention 2 is not available: {str(e)}")
    print("   Falling back to scaled dot product attention (SDPA)")
    print("   This may be slower but will still work correctly")

try:
    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Set attention implementation based on availability
    if use_flash_attention:
        config.llm_config._attn_implementation = 'flash_attention_2'
        attn_type = "flash_attention_2"
    else:
        # Use SDPA (scaled dot product attention) as fallback
        config.llm_config._attn_implementation = 'sdpa'
        attn_type = "scaled dot product attention (SDPA)"
    
    print(f"\nLoading model with {attn_type}...")
    try:
        model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
    except (ImportError, OSError, RuntimeError) as model_error:
        # If flash_attention_2 fails during model loading, retry with SDPA
        if use_flash_attention and 'flash' in str(model_error).lower():
            print(f"\n[WARNING] Flash Attention failed during model loading: {str(model_error)}")
            print("   Retrying with scaled dot product attention (SDPA)...")
            config.llm_config._attn_implementation = 'sdpa'
            model = AutoModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()
        else:
            raise
    
    # Move to GPU if available
    if torch.cuda.is_available():
        device_type = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        model = model.cuda()
    else:
        device_type = 'cpu'
        print("Using CPU (GPU not available)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    print("Model loaded successfully!")
    
    # Load test image
    test_image_path = Path("test.png")
    if not test_image_path.exists():
        print(f"\nError: {test_image_path} not found!")
        sys.exit(1)
    
    print(f"\nLoading test image: {test_image_path}")
    image = Image.open(test_image_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    # Set up generation config
    generation_config = dict(max_new_tokens=2048, do_sample=True)
    
    # Run OCR with autocast for optimal performance
    print("\n" + "-" * 60)
    print("Running OCR on test image (with autocast optimization)...")
    print("-" * 60)
    
    question = '<image>\nExtract all text from this image. Return only the extracted text, no additional commentary.'
    
    # Use autocast for Automatic Mixed-Precision (optimizes memory and performance)
    # This works with both Flash Attention 2 and SDPA
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        response, history = model.chat(
            tokenizer,
            str(test_image_path),
            question,
            generation_config,
            history=None,
            return_history=True
        )
    
    print("\n" + "=" * 60)
    print("OCR RESULT:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

