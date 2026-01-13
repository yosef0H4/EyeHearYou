"""
Test script for RapidOCR GPU support
Tests RapidOCR with GPU acceleration enabled

Usage:
    python test_rapidocr_gpu.py [--gpu] [--cpu]
    
    --gpu: Force GPU mode (default: check config.json)
    --cpu: Force CPU mode
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import time

# Parse arguments
parser = argparse.ArgumentParser(description="Test RapidOCR GPU support")
parser.add_argument("--gpu", action="store_true", help="Force GPU mode")
parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
args = parser.parse_args()

print("=" * 60)
print("RapidOCR GPU Test Script")
print("=" * 60)

try:
    import onnxruntime as ort
    from rapidocr_onnxruntime import RapidOCR
    
    # Check available providers
    print("\n" + "-" * 60)
    print("ONNX Runtime Providers")
    print("-" * 60)
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    
    has_cuda = 'CUDAExecutionProvider' in available_providers
    print(f"CUDA Execution Provider: {'Available' if has_cuda else 'Not Available'}")
    
    # Determine GPU usage
    if args.cpu:
        use_gpu = False
        print("\n[INFO] Forcing CPU mode (--cpu flag)")
    elif args.gpu:
        use_gpu = True
        print("\n[INFO] Forcing GPU mode (--gpu flag)")
    else:
        # Check config
        from src.backend.core.config import load_config
        config = load_config()
        td_config = config.get("text_detection", {})
        use_gpu = td_config.get("use_gpu", False)
        print(f"\n[INFO] Using config setting: use_gpu={use_gpu}")
    
    if use_gpu and not has_cuda:
        print("\n[WARNING] GPU requested but CUDAExecutionProvider not available!")
        print("          Falling back to CPU mode")
        use_gpu = False
    
    # Initialize RapidOCR
    print("\n" + "-" * 60)
    print("Initializing RapidOCR")
    print("-" * 60)
    
    device_str = "GPU" if use_gpu else "CPU"
    print(f"Device: {device_str}")
    
    try:
        # Try to pass providers if GPU is enabled
        if use_gpu:
            try:
                ocr = RapidOCR(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                print("[OK] RapidOCR initialized with CUDAExecutionProvider")
            except TypeError:
                # Providers parameter not supported, use default (should auto-detect)
                ocr = RapidOCR()
                print("[OK] RapidOCR initialized (should auto-detect GPU)")
        else:
            ocr = RapidOCR()
            print("[OK] RapidOCR initialized with CPU")
    except Exception as e:
        print(f"[ERROR] Failed to initialize RapidOCR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check for test image
    test_image_path = Path("test.png")
    if not test_image_path.exists():
        print(f"\n[ERROR] Test image not found: {test_image_path}")
        print("Please ensure test.png exists in the current directory")
        sys.exit(1)
    
    # Load test image
    print(f"\nLoading test image: {test_image_path}")
    test_image = Image.open(test_image_path)
    img_array = np.array(test_image)
    print(f"Image size: {test_image.size[0]}x{test_image.size[1]} pixels")
    print(f"Image mode: {test_image.mode}")
    
    # Run detection
    print("\n" + "-" * 60)
    print(f"Running text detection ({device_str})...")
    print("-" * 60)
    
    start_time = time.time()
    result, _ = ocr(img_array, use_det=True, use_rec=False)
    elapsed_time = time.time() - start_time
    
    if result:
        print(f"\n[OK] Detection completed in {elapsed_time:.3f} seconds")
        print(f"Found {len(result)} text regions")
        
        # Show first few regions
        print("\nFirst 5 detected regions:")
        for i, poly in enumerate(result[:5]):
            if isinstance(poly, list) and len(poly) >= 4:
                x_coords = [float(point[0]) for point in poly]
                y_coords = [float(point[1]) for point in poly]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                print(f"  Region {i+1}: ({x1}, {y1}) to ({x2}, {y2})")
    else:
        print("\n[WARNING] No text regions detected")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"[OK] Device: {device_str}")
    print(f"[OK] Detection time: {elapsed_time:.3f} seconds")
    print(f"[OK] Regions found: {len(result) if result else 0}")
    
    if use_gpu:
        print("\n[INFO] GPU acceleration is enabled")
        print("       Compare this speed with CPU mode (--cpu flag) to see improvement")
    else:
        print("\n[INFO] CPU mode is enabled")
        print("       Enable GPU in config.json or use --gpu flag to test GPU acceleration")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n[ERROR] Import Error: {e}")
    print("\nPlease install required packages:")
    print("  uv pip install rapidocr-onnxruntime")
    if not args.cpu:
        print("  uv pip install onnxruntime-gpu  # For GPU support")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

