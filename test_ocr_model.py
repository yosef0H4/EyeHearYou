"""
Test script for H2OVL OCR Model
Tests the OCR model to verify installation and functionality

Usage:
    python test_ocr_model.py [--no-patch]
    
    --no-patch: Skip applying the torchvision compatibility patch
"""
import argparse
import sys
from pathlib import Path
import importlib.util

# Parse arguments
parser = argparse.ArgumentParser(description="Test H2OVL OCR Model installation")
parser.add_argument("--no-patch", action="store_true", help="Skip applying the torchvision compatibility patch")
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

import warnings
# Suppress informational warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*Flash Attention 2 only supports.*")
warnings.filterwarnings("ignore", message=".*Flash Attention is not available.*")
warnings.filterwarnings("ignore", message=".*FlashAttention is not installed.*")

print("=" * 60)
print("H2OVL OCR Model Test Script")
print("=" * 60)

try:
    import torch
    from PIL import Image
    
    # Print system information
    print("\n" + "-" * 60)
    print("System Information")
    print("-" * 60)
    
    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA information
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device_type = 'cuda'
    else:
        print("CUDA Available: No")
        print("Using CPU (GPU not available)")
        device_type = 'cpu'
    
    # Flash Attention check
    print("\nFlash Attention Status:")
    flash_attention_available = False
    flash_attention_version = None
    try:
        import flash_attn
        flash_attention_available = True
        try:
            flash_attention_version = flash_attn.__version__
            print(f"  Flash Attention 2: Available (version {flash_attention_version})")
            flash_attn_status = f"Enabled (version {flash_attention_version})"
        except AttributeError:
            print(f"  Flash Attention 2: Available")
            flash_attn_status = "Enabled"
    except (ImportError, OSError) as e:
        print(f"  Flash Attention 2: Not available")
        print(f"  Fallback: Using SDPA (Scaled Dot Product Attention)")
        flash_attn_status = "Disabled (SDPA)"
    
    # Check for test image
    test_image_path = Path("test.png")
    if not test_image_path.exists():
        print(f"\n[ERROR] Test image not found: {test_image_path}")
        print("Please ensure test.png exists in the current directory")
        sys.exit(1)
    
    print("\n" + "-" * 60)
    print("Loading OCR Model")
    print("-" * 60)
    print("Model: H2OVL-Mississippi-0.8B (h2oai/h2ovl-mississippi-800m)")
    print("(This will download the model on first run - may take a few minutes)")
    
    # Import model loader
    sys.path.insert(0, str(Path(__file__).parent))
    from src.backend.core.model_loader import get_model
    
    # Load model
    print("\nInitializing model...")
    model = get_model()
    print("[OK] Model initialized successfully!")
    
    # Load test image
    print(f"\nLoading test image: {test_image_path}")
    test_image = Image.open(test_image_path)
    print(f"Image size: {test_image.size[0]}x{test_image.size[1]} pixels")
    print(f"Image mode: {test_image.mode}")
    
    # Run OCR
    print("\n" + "-" * 60)
    print("Running OCR on test image...")
    print("-" * 60)
    
    extracted_text = model.predict(test_image)
    
    if extracted_text:
        print("\n[OK] OCR completed successfully!")
        print("\nExtracted Text:")
        print("-" * 60)
        print(extracted_text)
        print("-" * 60)
        print(f"\nText length: {len(extracted_text)} characters")
    else:
        print("\n[WARNING] OCR returned empty result")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✓ Model loaded successfully")
    print(f"✓ Image processed: {test_image_path}")
    print(f"✓ Text extracted: {len(extracted_text)} characters")
    print(f"✓ Device: {device_type.upper()}")
    print(f"✓ Flash Attention: {flash_attn_status}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n[ERROR] Import Error: {e}")
    print("\nPlease install required packages:")
    print("  uv pip install torch transformers pillow")
    print("  (Optional) uv pip install flash-attn --no-build-isolation")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

