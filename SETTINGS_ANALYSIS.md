# Settings Analysis: What Triggers RapidOCR vs Post-Processing

## Summary
Currently, **ALL settings trigger a full RapidOCR re-run**, but many should only re-filter cached results.

---

## ✅ NEEDS RapidOCR (Affects Image Before Detection)

These settings modify the image itself, so RapidOCR must re-run:

### Preprocessing Settings (lines 272, 275, 277, 314 in window.py)
1. **`binary_threshold`** (0-255)
   - Applies thresholding to image
   - **MUST re-run RapidOCR** ✅

2. **`invert`** (True/False)
   - Inverts image colors
   - **MUST re-run RapidOCR** ✅

3. **`dilation`** (0-5)
   - Thickens text in image
   - **MUST re-run RapidOCR** ✅

4. **`contrast`** (0.5-3.0)
   - Adjusts image contrast
   - **MUST re-run RapidOCR** ✅

5. **`brightness`** (-100 to 100)
   - Adjusts image brightness
   - **MUST re-run RapidOCR** ✅

### Image Resize (lines 214, 215 in window.py)
6. **`max_image_dimension`** (320-2560)
   - Resizes image before processing
   - **MUST re-run RapidOCR** ✅

---

## ❌ POST-PROCESSING ONLY (Should Use Cache)

These settings only filter/reorder existing detections. They should NOT trigger RapidOCR:

### Detection Filter Settings (lines 398, 400, 424, 426 in window.py)
1. **`min_width`** (5-1000 px)
   - Filters boxes by minimum width
   - **POST-PROCESSING ONLY** ❌ (Currently wastes time re-running RapidOCR)
   - Used in: `filter_text_regions()` at line 157 of `filtering.py`

2. **`min_height`** (5-1000 px)
   - Filters boxes by minimum height
   - **POST-PROCESSING ONLY** ❌ (Currently wastes time re-running RapidOCR)
   - Used in: `filter_text_regions()` at line 157 of `filtering.py`

### Merging Settings (lines 514, 515 in window.py)
3. **`merge_vertical_tolerance`** (0-300 px)
   - Controls vertical gap for merging lines
   - **POST-PROCESSING ONLY** ❌ (Currently wastes time re-running RapidOCR)
   - Used in: `merge_close_text_boxes()` at line 130 of `merging.py`

4. **`merge_horizontal_tolerance`** (0-300 px)
   - Controls horizontal gap for merging words
   - **POST-PROCESSING ONLY** ❌ (Currently wastes time re-running RapidOCR)
   - Used in: `merge_close_text_boxes()` at line 130 of `merging.py`

5. **`merge_width_ratio_threshold`** (0.0-1.0)
   - Width ratio for merge validation
   - **POST-PROCESSING ONLY** ❌ (Currently wastes time re-running RapidOCR)
   - Used in: `merge_close_text_boxes()` at line 130 of `merging.py`

### Sorting Settings (lines 336, 548, 550 in window.py)
6. **`direction`** (reading order: horizontal_ltr, horizontal_rtl, vertical_ltr, vertical_rtl)
   - Changes sort order of detected boxes
   - **POST-PROCESSING ONLY** ❌ (Currently wastes time re-running RapidOCR)
   - Used in: `sort_text_regions_by_reading_order()` in `filtering.py`

7. **`group_tolerance`** (0.1-2.0)
   - Multiplier for line/column grouping
   - **POST-PROCESSING ONLY** ❌ (Currently wastes time re-running RapidOCR)
   - Used in: `sort_text_regions_by_reading_order()` in `filtering.py`

---

## Current Flow (Inefficient)

```
User changes min_width/min_height
  ↓
save_and_refresh() called
  ↓
run_detection_preview() called
  ↓
start_worker() called
  ↓
process_image() - Preprocessing (OK, but unnecessary if only post-processing changed)
  ↓
detect_text_regions_unfiltered() - **RAPIDOCR RUNS** ❌ (Wasteful!)
  ↓
filter_text_regions() - Applies min_width/min_height
  ↓
sort_text_regions_by_reading_order() - Applies direction/group_tolerance
  ↓
merge_close_text_boxes() - Applies merge settings
```

---

## Optimal Flow (Efficient)

```
User changes min_width/min_height
  ↓
Check: Did preprocessing change? NO
  ↓
Check: Is raw detection cache available? YES
  ↓
Skip RapidOCR ❌
  ↓
filter_text_regions() - Apply new min_width/min_height to cached results
  ↓
sort_text_regions_by_reading_order() - Apply direction/group_tolerance
  ↓
merge_close_text_boxes() - Apply merge settings
```

---

## Code Locations

### Settings that trigger refresh:
- **Preprocessing**: `window.py` lines 272, 275, 277, 314
- **Detection filters**: `window.py` lines 398, 400, 424, 426
- **Merging**: `window.py` lines 514, 515
- **Sorting**: `window.py` lines 336, 548, 550
- **Resize**: `window.py` lines 214, 215

### Processing pipeline:
- **Worker entry**: `worker.py` line 26 (`run()` method)
- **Preprocessing**: `worker.py` line 38 (`process_image()`)
- **RapidOCR call**: `worker.py` line 51 (`detect_text_regions_unfiltered()`)
- **Filtering**: `worker.py` line 83 (`filter_text_regions()`)
- **Sorting**: `worker.py` line 105 (`sort_text_regions_by_reading_order()`)
- **Merging**: `worker.py` line 129 (`merge_close_text_boxes()`)

### Cache infrastructure:
- **Cache storage**: `state.py` line 18 (`cached_raw_detections`)
- **Cache key**: Uses `screenshot_version` (line 14 in `state.py`)

---

## Recommendation

Implement smart caching:
1. Cache raw detections keyed by: `(screenshot_version, preprocessing_hash)`
2. When settings change, check if preprocessing changed
3. If only post-processing changed → use cache, skip RapidOCR
4. If preprocessing changed → clear cache, run RapidOCR

This would make changing `min_width`, `min_height`, merge settings, or sorting settings **instant** instead of re-running RapidOCR every time.

