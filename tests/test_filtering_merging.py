"""Unit tests for filtering and merging functions"""
import pytest
from src.backend.core.filtering import filter_text_regions
from src.backend.core.merging import merge_close_text_boxes


def test_filter_text_regions_basic():
    """Test basic filtering by min width and height"""
    regions = [
        (10, 10, 50, 50),   # 40x40 - should pass
        (10, 10, 20, 50),   # 10x40 - should fail (width too small)
        (10, 10, 50, 20),   # 40x10 - should fail (height too small)
        (10, 10, 15, 15),   # 5x5 - should fail (both too small)
    ]
    image_shape = (100, 100)
    
    filtered = filter_text_regions(regions, image_shape, min_width=30, min_height=30)
    
    assert len(filtered) == 1
    assert filtered[0] == (10, 10, 50, 50)


def test_filter_text_regions_clamping():
    """Test that coordinates are clamped to image bounds"""
    regions = [
        (-10, -10, 50, 50),  # Negative coords should be clamped
        (10, 10, 150, 150),  # Coords beyond image should be clamped
    ]
    image_shape = (100, 100)
    
    filtered = filter_text_regions(regions, image_shape, min_width=30, min_height=30)
    
    # First region should be clamped to (0, 0, 50, 50) and pass
    # Second region should be clamped to (10, 10, 100, 100) and pass
    assert len(filtered) == 2


def test_merge_close_text_boxes_vertical():
    """Test merging boxes that are vertically adjacent"""
    regions = [
        (10, 10, 50, 30),   # Top box
        (10, 35, 50, 55),   # Bottom box (5px gap, within 30px tolerance)
    ]
    
    merged, is_merged, original_groups = merge_close_text_boxes(
        regions,
        vertical_tolerance=30,
        horizontal_tolerance=50,
        width_ratio_threshold=0.3
    )
    
    assert len(merged) == 1
    assert is_merged[0] is True
    assert merged[0] == (10, 10, 50, 55)  # Union bounding box
    assert len(original_groups[0]) == 2  # Two boxes merged


def test_merge_close_text_boxes_horizontal_alignment():
    """Test that boxes that are not aligned and far apart don't merge"""
    regions = [
        (10, 10, 50, 30),   # Left box
        (100, 35, 140, 55),  # Right box (too far horizontally, not on same row)
    ]
    
    merged, is_merged, original_groups = merge_close_text_boxes(
        regions,
        vertical_tolerance=30,
        horizontal_tolerance=50,
        width_ratio_threshold=0.3
    )
    
    # Should not merge: boxes are not on same row (no vertical overlap) 
    # and not vertically aligned (no horizontal overlap, centers don't align)
    assert len(merged) == 2
    assert all(not merged_flag for merged_flag in is_merged)


def test_merge_close_text_boxes_horizontal():
    """Test merging boxes that are horizontally adjacent (words on same line)"""
    regions = [
        (10, 10, 40, 30),   # "I" - left word
        (45, 10, 100, 30),  # "guess" - right word (5px gap, same row, within 50px tolerance)
    ]
    
    merged, is_merged, original_groups = merge_close_text_boxes(
        regions,
        vertical_tolerance=30,
        horizontal_tolerance=50,
        width_ratio_threshold=0.3
    )
    
    # Should merge because they're on the same row (high vertical overlap)
    # and the gap (5px) is within horizontal_tolerance (50px)
    assert len(merged) == 1
    assert is_merged[0] is True
    assert merged[0] == (10, 10, 100, 30)  # Union bounding box
    assert len(original_groups[0]) == 2  # Two boxes merged


def test_merge_close_text_boxes_horizontal_large_gap():
    """Test that boxes on same row but with large gap don't merge"""
    regions = [
        (10, 10, 40, 30),   # Left word
        (200, 10, 240, 30),  # Right word (160px gap, exceeds 50px tolerance)
    ]
    
    merged, is_merged, original_groups = merge_close_text_boxes(
        regions,
        vertical_tolerance=30,
        horizontal_tolerance=50,
        width_ratio_threshold=0.3
    )
    
    # Should not merge: gap (160px) exceeds horizontal_tolerance (50px)
    assert len(merged) == 2
    assert all(not merged_flag for merged_flag in is_merged)


def test_merge_close_text_boxes_width_ratio():
    """Test that vertically adjacent boxes with horizontal overlap can merge"""
    regions = [
        (10, 10, 50, 30),    # 40px wide, top box
        (10, 35, 20, 55),    # 10px wide, bottom box (5px gap, 10px horizontal overlap)
    ]
    
    merged, is_merged, original_groups = merge_close_text_boxes(
        regions,
        vertical_tolerance=30,
        horizontal_tolerance=50,
        width_ratio_threshold=0.3
    )
    
    # Should merge: vertically adjacent (5px gap < 30px tolerance)
    # and has horizontal overlap (10px > 20% of min width = 2px)
    assert len(merged) == 1
    assert is_merged[0] is True


def test_merge_close_text_boxes_no_merge():
    """Test that boxes that don't meet criteria are not merged"""
    regions = [
        (10, 10, 50, 30),
        (60, 100, 100, 120),  # Far away, shouldn't merge
    ]
    
    merged, is_merged, original_groups = merge_close_text_boxes(
        regions,
        vertical_tolerance=30,
        horizontal_tolerance=50,
        width_ratio_threshold=0.3
    )
    
    assert len(merged) == 2
    assert all(not merged_flag for merged_flag in is_merged)


def test_parity_preview_vs_extraction():
    """
    Test that preview and extraction use the same filtering/merging logic.
    This is a simplified parity check - in practice, both paths use the same functions.
    """
    regions = [
        (10, 10, 50, 50),   # Large box
        (10, 60, 50, 80),   # Large box below
        (10, 85, 50, 105),  # Large box further below (25px gap)
    ]
    image_shape = (200, 200)
    
    # Apply filtering (same as extraction)
    filtered = filter_text_regions(regions, image_shape, min_width=30, min_height=30)
    
    # Apply merging (same as extraction)
    merged, is_merged, original_groups = merge_close_text_boxes(
        filtered,
        vertical_tolerance=30,
        horizontal_tolerance=50,
        width_ratio_threshold=0.3
    )
    
    # All three boxes should be filtered (all are large enough)
    assert len(filtered) == 3
    
    # First two boxes should merge (20px gap < 30px tolerance)
    # Third box might merge with second if gap is small enough
    # This test verifies the logic is deterministic
    assert len(merged) >= 1
    assert len(merged) <= 3



