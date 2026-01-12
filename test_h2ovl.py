"""
Test script for H2OVL-Mississippi-0.8B OCR model
Tests the model against test.png to verify installation and functionality
"""
import warnings
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from PIL import Image
import sys
from pathlib import Path

# Suppress informational warnings (optional - remove if you want to see all warnings)
# Note: Some warnings appear during model loading (before this code runs) and cannot be suppressed here
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*Flash Attention 2 only supports.*")
# Autocast is used during inference to optimize performance and address Flash Attention 2 recommendations

# Set up the model and tokenizer
model_path = 'h2oai/h2ovl-mississippi-800m'

print("=" * 60)
print("H2OVL-Mississippi-0.8B Test Script")
print("=" * 60)
print(f"\nLoading model from: {model_path}")
print("This may take a minute on first run (downloading model)...")

try:
    # Load config with flash attention
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.llm_config._attn_implementation = 'flash_attention_2'
    
    print("\nLoading model with flash_attention_2...")
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    
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
    
    # Run OCR with autocast for optimal Flash Attention 2 performance
    print("\n" + "-" * 60)
    print("Running OCR on test image (with autocast optimization)...")
    print("-" * 60)
    
    question = '<image>\nExtract all text from this image. Return only the extracted text, no additional commentary.'
    
    # Use autocast for Automatic Mixed-Precision (recommended for Flash Attention 2)
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

