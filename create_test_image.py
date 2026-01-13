"""
Create a copyright-free synthetic OCR test image
This generates a simple image with text for testing OCR functionality
"""
from PIL import Image, ImageDraw, ImageFont
import sys
from pathlib import Path

def create_synthetic_test_image(output_path="test.png", width=1920, height=1080):
    """
    Create a synthetic test image with text for OCR testing
    
    Args:
        output_path: Where to save the image
        width: Image width in pixels
        height: Image height in pixels
    """
    print("=" * 60)
    print("Creating Synthetic OCR Test Image")
    print("=" * 60)
    
    # Create a simple background (light gray)
    img = Image.new('RGB', (width, height), color='#F5F5F5')
    draw = ImageDraw.Draw(img)
    
    # Add some text blocks (simulating game dialogue)
    test_texts = [
        "Hello, this is a test image for OCR.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing text extraction capabilities.",
        "This image is copyright-free and safe to use.",
        "1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ]
    
    # Try to use a default font, fallback to basic if not available
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
        except:
            # Fallback to default font
            print("Warning: Using default font (may not render perfectly)")
            font = ImageFont.load_default()
    
    # Draw text blocks
    y_position = 100
    for i, text in enumerate(test_texts):
        # Draw text with slight shadow for better visibility
        draw.text((110, y_position + 2), text, fill='#333333', font=font)
        draw.text((108, y_position), text, fill='#000000', font=font)
        y_position += 80
    
    # Add a darker box at the bottom (simulating dialogue box)
    box_y = height - 200
    draw.rectangle([50, box_y, width - 50, height - 50], fill='#2C2C2C', outline='#000000', width=2)
    
    # Add text in the box (white text on dark background)
    box_text = "This is a synthetic test image for OCR testing."
    try:
        box_font = ImageFont.truetype("arial.ttf", 36)
    except:
        try:
            box_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 36)
        except:
            box_font = ImageFont.load_default()
    
    draw.text((70, box_y + 30), box_text, fill='#FFFFFF', font=box_font)
    
    # Save the image
    img.save(output_path, 'PNG')
    print(f"\n[OK] Successfully created: {output_path}")
    print(f"  Image size: {width}x{height} pixels")
    print(f"  This image is copyright-free and safe to use.")
    
    return True


def main():
    """Main function"""
    output_path = "test.png"
    
    # Check if test.png already exists
    if Path(output_path).exists():
        print(f"\nNote: {output_path} already exists. Overwriting...")
    
    create_synthetic_test_image(output_path)
    
    print("\n" + "=" * 60)
    print("Alternative: Download from Public Domain Sources")
    print("=" * 60)
    print("\nIf you prefer a real-world test image, you can download from:")
    print("\n1. Wikimedia Commons - Test OCR Document:")
    print("   https://commons.wikimedia.org/wiki/File:Test_OCR_document.jpg")
    print("   (Click 'Download' button, then rename to test.png)")
    print("\n2. Matt Mahoney's OCR Test Images:")
    print("   https://www.mattmahoney.net/ocr/")
    print("   (Download any image, rename to test.png)")
    print("\nAll of these are in the public domain and safe to use.")


if __name__ == "__main__":
    main()

