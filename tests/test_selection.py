"""Comprehensive tests for selection functionality"""
import pytest
import numpy as np
from PIL import Image
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.backend.core.filtering import (
    generate_selection_mask,
    filter_regions_by_mask,
    get_regions_from_mask
)
from src.backend.state import state


def test_generate_selection_mask_default_select_all():
    """Test default state: everything selected (base_state=True, no ops)"""
    image_shape = (100, 200)  # height, width
    mask = generate_selection_mask(image_shape, [], base_state=True)
    
    assert mask is not None
    assert mask.shape == (100, 200)
    assert np.all(mask == 255)  # All white (selected)


def test_generate_selection_mask_default_deselect_all():
    """Test default state: nothing selected (base_state=False, no ops)"""
    image_shape = (100, 200)
    mask = generate_selection_mask(image_shape, [], base_state=False)
    
    assert mask is not None
    assert mask.shape == (100, 200)
    assert np.all(mask == 0)  # All black (deselected)


def test_generate_selection_mask_add_region():
    """Test adding a selection region"""
    image_shape = (100, 200)
    # Add a region: x=0.1, y=0.1, w=0.3, h=0.4 (normalized)
    ops = [{"op": "add", "rect": (0.1, 0.1, 0.3, 0.4)}]
    mask = generate_selection_mask(image_shape, ops, base_state=False)
    
    assert mask is not None
    # Convert normalized to pixels: x=20, y=10, w=60, h=40
    # So region from (20, 10) to (80, 50) should be white
    assert mask[10, 20] == 255  # Top-left corner of added region
    assert mask[49, 79] == 255  # Bottom-right corner of added region
    assert mask[0, 0] == 0  # Outside region should be black


def test_generate_selection_mask_subtract_region():
    """Test subtracting a selection region"""
    image_shape = (100, 200)
    # Start with all selected, then subtract a region
    ops = [{"op": "sub", "rect": (0.1, 0.1, 0.3, 0.4)}]
    mask = generate_selection_mask(image_shape, ops, base_state=True)
    
    assert mask is not None
    # The subtracted region should be black
    assert mask[10, 20] == 0  # Inside subtracted region
    assert mask[0, 0] == 255  # Outside should still be white


def test_generate_selection_mask_multiple_ops():
    """Test multiple operations (add then subtract)"""
    image_shape = (100, 200)
    ops = [
        {"op": "add", "rect": (0.0, 0.0, 0.5, 0.5)},  # Add left half
        {"op": "sub", "rect": (0.1, 0.1, 0.2, 0.2)}  # Subtract small region
    ]
    mask = generate_selection_mask(image_shape, ops, base_state=False)
    
    assert mask is not None
    # Left half should be mostly white, except subtracted area
    assert mask[5, 10] == 255  # In added region, not in subtracted
    assert mask[15, 30] == 0  # In subtracted region


def test_generate_selection_mask_normalized_coords():
    """Test that normalized coordinates work correctly"""
    image_shape = (50, 100)
    # Normalized: (0.2, 0.3, 0.4, 0.3)
    # Pixels: x=20, y=15, w=40, h=15
    ops = [{"op": "add", "rect": (0.2, 0.3, 0.4, 0.3)}]
    mask = generate_selection_mask(image_shape, ops, base_state=False)
    
    assert mask is not None
    assert mask[15, 20] == 255  # Top-left
    assert mask[29, 59] == 255  # Bottom-right (15+15-1, 20+40-1)
    assert mask[14, 19] == 0  # Just outside


def test_generate_selection_mask_clamping():
    """Test that coordinates are clamped to image bounds"""
    image_shape = (100, 200)
    # Region that extends beyond image bounds
    ops = [{"op": "add", "rect": (0.9, 0.9, 0.2, 0.2)}]  # Would go out of bounds
    mask = generate_selection_mask(image_shape, ops, base_state=False)
    
    assert mask is not None
    # Should not crash, and should clamp properly
    assert mask.shape == (100, 200)


def test_filter_regions_by_mask_keep_all():
    """Test filtering when mask keeps all regions"""
    regions = [(10, 10, 50, 50), (60, 60, 100, 100)]
    mask = np.full((200, 200), 255, dtype=np.uint8)  # All white
    
    filtered = filter_regions_by_mask(regions, mask)
    assert len(filtered) == 2
    assert filtered == regions


def test_filter_regions_by_mask_discard_all():
    """Test filtering when mask discards all regions"""
    regions = [(10, 10, 50, 50), (60, 60, 100, 100)]
    mask = np.zeros((200, 200), dtype=np.uint8)  # All black
    
    filtered = filter_regions_by_mask(regions, mask)
    assert len(filtered) == 0


def test_filter_regions_by_mask_partial():
    """Test filtering when mask keeps some regions"""
    regions = [(10, 10, 50, 50), (60, 60, 100, 100)]
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Make first region white
    mask[10:50, 10:50] = 255
    
    filtered = filter_regions_by_mask(regions, mask)
    assert len(filtered) == 1
    assert filtered[0] == regions[0]


def test_filter_regions_by_mask_threshold():
    """Test filtering with threshold (partial overlap)"""
    regions = [(10, 10, 50, 50)]
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Make 20% of region white (should be kept with default 10% threshold)
    mask[10:30, 10:30] = 255
    
    filtered = filter_regions_by_mask(regions, mask, threshold=0.15)
    assert len(filtered) == 1  # 20% > 15% threshold, so kept
    
    # With higher threshold, should be discarded
    filtered2 = filter_regions_by_mask(regions, mask, threshold=0.25)
    assert len(filtered2) == 0  # 20% < 25% threshold, so discarded


def test_get_regions_from_mask_simple():
    """Test extracting regions from a simple mask"""
    mask = np.zeros((100, 200), dtype=np.uint8)
    # Create a white rectangle
    mask[20:60, 30:80] = 255
    
    regions = get_regions_from_mask(mask)
    assert len(regions) == 1
    x1, y1, x2, y2 = regions[0]
    assert x1 == 30
    assert y1 == 20
    assert x2 == 80
    assert y2 == 60


def test_get_regions_from_mask_multiple():
    """Test extracting multiple regions from mask"""
    mask = np.zeros((100, 200), dtype=np.uint8)
    # Create two separate white regions
    mask[10:30, 10:50] = 255
    mask[50:70, 100:150] = 255
    
    regions = get_regions_from_mask(mask)
    assert len(regions) == 2
    # Regions should be in some order
    coords = [tuple(r) for r in regions]
    assert (10, 10, 50, 30) in coords
    assert (100, 50, 150, 70) in coords


def test_get_regions_from_mask_min_size():
    """Test that very small regions are filtered out"""
    mask = np.zeros((100, 200), dtype=np.uint8)
    # Create a very small region (2x2 pixels, should be filtered)
    mask[10:12, 10:12] = 255
    
    regions = get_regions_from_mask(mask)
    # Should be filtered out (min size is 4x4)
    assert len(regions) == 0


def test_selection_state_defaults():
    """Test that state has correct defaults"""
    # Reset state
    state.use_rapidocr = True
    state.selection_base_state = True
    state.selection_ops = []
    state.manual_boxes = []
    
    assert state.use_rapidocr is True
    assert state.selection_base_state is True
    assert state.selection_ops == []
    assert state.manual_boxes == []


def test_selection_workflow_integration():
    """Test a complete workflow: deselect all, then add specific region"""
    image_shape = (100, 200)
    
    # Step 1: Deselect all
    mask1 = generate_selection_mask(image_shape, [], base_state=False)
    assert np.all(mask1 == 0)
    
    # Step 2: Add a region (dialog box at bottom)
    ops = [{"op": "add", "rect": (0.1, 0.7, 0.8, 0.25)}]  # Bottom 25% of image
    mask2 = generate_selection_mask(image_shape, ops, base_state=False)
    
    # Bottom region should be white
    assert mask2[70, 20] == 255  # In added region
    assert mask2[0, 0] == 0  # Top should be black
    
    # Step 3: Filter some text regions
    regions = [
        (20, 10, 50, 30),   # Top region (should be filtered out)
        (20, 75, 180, 95),  # Bottom region (should be kept)
    ]
    filtered = filter_regions_by_mask(regions, mask2)
    assert len(filtered) == 1
    assert filtered[0] == regions[1]  # Only bottom region kept


def test_selection_with_test_image():
    """Test selection with actual test.png if available"""
    test_image_path = "test.png"
    if not os.path.exists(test_image_path):
        pytest.skip("test.png not found")
    
    img = Image.open(test_image_path)
    img_h, img_w = img.size[1], img.size[0]
    
    # Test: Select only bottom 30% (where dialog usually is)
    ops = [{"op": "add", "rect": (0.0, 0.7, 1.0, 0.3)}]
    mask = generate_selection_mask((img_h, img_w), ops, base_state=False)
    
    assert mask is not None
    assert mask.shape == (img_h, img_w)
    
    # Bottom should be white
    assert mask[img_h - 10, img_w // 2] == 255
    # Top should be black
    assert mask[10, img_w // 2] == 0


def test_manual_boxes_integration():
    """Test that manual boxes work with state"""
    state.manual_boxes = []
    
    # Add some manual boxes
    state.manual_boxes.append((10, 10, 50, 50))
    state.manual_boxes.append((100, 100, 150, 150))
    
    assert len(state.manual_boxes) == 2
    assert state.manual_boxes[0] == (10, 10, 50, 50)


def test_selection_mask_edge_cases():
    """Test edge cases for selection masks"""
    image_shape = (100, 200)
    
    # Empty ops list
    mask = generate_selection_mask(image_shape, [], base_state=True)
    assert np.all(mask == 255)
    
    # Ops with zero width/height (should be ignored)
    ops = [{"op": "add", "rect": (0.5, 0.5, 0.0, 0.1)}]
    mask = generate_selection_mask(image_shape, ops, base_state=False)
    # Should not crash, and zero-width rect should be ignored
    
    # Ops outside image bounds (should be clamped)
    ops = [{"op": "add", "rect": (-0.1, -0.1, 1.2, 1.2)}]
    mask = generate_selection_mask(image_shape, ops, base_state=False)
    assert mask is not None
    assert mask.shape == (100, 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

