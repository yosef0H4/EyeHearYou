#!/usr/bin/env python3
"""
OCR Script using Ovis2.5-2B model
Takes image paths as input and returns the extracted text.
"""

import sys
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from pathlib import Path

# Workaround for triton on Windows - create a stub if triton is not available
try:
    import triton
    import triton.language
except ImportError:
    # Create a minimal triton stub for Windows compatibility
    import importlib.util
    from types import ModuleType
    
    # Mock decorator/function classes
    class MockJIT:
        def __call__(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    
    class MockAutotune:
        def __call__(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    
    # Create main triton module
    triton_module = ModuleType('triton')
    triton_module.__file__ = '<stub>'
    spec = importlib.util.spec_from_loader('triton', loader=None)
    triton_module.__spec__ = spec
    triton_module.jit = MockJIT()
    triton_module.autotune = MockAutotune()
    sys.modules['triton'] = triton_module
    
    # Create triton.language submodule with common attributes
    triton_language = ModuleType('triton.language')
    triton_language.__file__ = '<stub>'
    spec_lang = importlib.util.spec_from_loader('triton.language', loader=None)
    triton_language.__spec__ = spec_lang
    
    # Add common triton.language attributes
    triton_language.constexpr = lambda x: x
    triton_language.static = lambda x: x
    
    sys.modules['triton.language'] = triton_language
    
    # Add language as attribute to main module
    triton_module.language = triton_language
    
    print("Note: Using triton stub (triton not available on Windows)")


def extract_text_from_image(image_path: str, model, enable_thinking: bool = False) -> str:
    """
    Extract text from an image using the Ovis2.5-2B model.
    
    Args:
        image_path: Path to the image file
        model: Loaded Ovis2.5-2B model
        enable_thinking: Whether to enable thinking mode for better accuracy
    
    Returns:
        Extracted text from the image
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    
    # Create the input message
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract all text from this image. End your response with 'Final answer: '."},
        ],
    }]
    
    # Preprocess inputs
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    # Move inputs to CUDA
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda() if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None
    
    # Generate output
    outputs = model.generate(
        inputs=input_ids,
        pixel_values=pixel_values,
        grid_thws=grid_thws,
        enable_thinking=enable_thinking,
        max_new_tokens=1024,
    )
    
    # Decode and return the response
    response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    """Main function to process images from command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python ocr_script.py <image_path1> [image_path2] ...")
        print("Example: python ocr_script.py image1.jpg image2.png")
        sys.exit(1)
    
    # Check for CUDA availability - required for this script
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a CUDA-capable GPU.")
        print("Please ensure you have:")
        print("  1. An NVIDIA GPU with CUDA support")
        print("  2. CUDA drivers installed")
        print("  3. PyTorch with CUDA support")
        sys.exit(1)
    
    # Display GPU information
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    print(f"GPU detected: {gpu_name}")
    print(f"GPU memory: {gpu_memory:.2f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    print()
    
    # Load the model
    MODEL_PATH = "AIDC-AI/Ovis2.5-2B"
    print(f"Loading model from {MODEL_PATH}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model = model.cuda()
        print(f"Model loaded successfully on GPU!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Process each image
    for image_path in sys.argv[1:]:
        if not Path(image_path).exists():
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")
        
        try:
            extracted_text = extract_text_from_image(image_path, model)
            print(f"\nExtracted Text:\n{extracted_text}\n")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue


if __name__ == "__main__":
    main()

