#!/usr/bin/env python3

import os
import re
import sys
import math
import json
import time
import warnings
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from datetime import datetime, timedelta
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds, reproject
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor

# CuPy for GPU-based processing
try:
    import cupy as cp
    print(f"[INFO] Using CuPy version: {cp.__version__}")

    # Check CUDA version
    cuda_version = cp.cuda.runtime.runtimeGetVersion()
    major = cuda_version // 1000
    minor = (cuda_version % 1000) // 10
    print(f"[INFO] CUDA version: {major}.{minor}")


    # Get GPU details
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props["name"].decode('utf-8')
    mem_total = device.mem_info[1] / (1024**3)
    mem_free = device.mem_info[0] / (1024**3)

    print(
        f"[INFO] GPU: {gpu_name} ({mem_free:.1f}GB free / {mem_total:.1f}GB total)")

    # Configure memory settings based on available GPU memory
    if mem_total < 4.0:
        # Small GPU (<4GB)
        GPU_CHUNK_SIZE = 1024
        GPU_MEMORY_LIMIT = "95%"
        print("[INFO] Small GPU detected - using very conservative settings")
    elif mem_total < 8.0:
        # Medium GPU (4-8GB)
        GPU_CHUNK_SIZE = 1024
        GPU_MEMORY_LIMIT = "75%"
        print("[INFO] Medium GPU detected - using conservative settings")
    else:
        # Large GPU (>8GB)
        GPU_CHUNK_SIZE = 2048
        GPU_MEMORY_LIMIT = "98%"
        print("[INFO] Large GPU detected - using balanced settings")

    # Configure environment variables
    os.environ["CUPY_GPU_MEMORY_LIMIT"] = GPU_MEMORY_LIMIT

    # Verify GPU is working
    print("[INFO] Testing GPU computation...")
    test_size = min(2000, int(2000 * (mem_free / 4.0)))
    a = cp.ones((test_size, test_size), dtype=cp.float32)
    b = cp.ones((test_size, test_size), dtype=cp.float32)
    c = cp.matmul(a, b)
    cp.cuda.Device().synchronize()
    del a, b, c
    cp._default_memory_pool.free_all_blocks()
    print("[INFO] GPU test successful")

    # GPU is available
    USE_GPU = True

except ImportError as e:
    print(f"[WARNING] GPU imports failed: {str(e)}")
    print("[WARNING] Falling back to CPU processing")
    USE_GPU = False
    GPU_CHUNK_SIZE = 512  # Default for CPU
except Exception as e:
    print(f"[WARNING] GPU initialization error: {str(e)}")
    print("[WARNING] Falling back to CPU processing")
    USE_GPU = False
    GPU_CHUNK_SIZE = 512  # Default for CPU

###############################################################################
# Configuration
###############################################################################

# Invalid SCL classes to discard
# Label | Classification
# ----------------------
# 0     | NO_DATA
# 1     | SATURATED_OR_DEFECTIVE
# 2     | DARK_AREA_PIXELS
# 3     | CLOUD_SHADOWS
# 4     | VEGETATION
# 5     | NOT_VEGETATED
# 6     | WATER
# 7     | UNCLASSIFIED
# 8     | CLOUD_MEDIUM_PROBABILITY
# 9     | CLOUD_HIGH_PROBABILITY
# 10    | THIN_CIRRUS
# 11    | SNOW
INVALID_SCL_CLASSES = {1, 3, 8, 9, 10, 11, 20}

# Regex for Sentinel-2 L2A file naming, e.g. S2B_15SUA_20240121_0_L2A_B04.tif
FILENAME_REGEX = re.compile(r"S2[ABC]_(\d{2}[A-Z]{3})_(\d{8})_(\d+)_L2A_")

# Mosaic resolution (CRS loaded from config)
TARGET_CRS = None  # Loaded from config.json
TARGET_RES_X = 10.0
TARGET_RES_Y = 10.0

# Processing options
PERCENTILE_VALUE = 25.0  # Percentile value for compositing

# Globals updated at runtime
RGB_DTYPE = None
RGB_NODATA = None

###############################################################################
# Core GPU percentile computation
###############################################################################

# Global GPU resources for reuse across chunks (reduces allocation overhead)
_gpu_streams = None
_pinned_buffers = {}

def init_gpu_resources():
    """Initialize reusable GPU resources for better performance."""
    global _gpu_streams, _pinned_buffers
    if USE_GPU and _gpu_streams is None:
        # Create CUDA streams for overlapping operations
        _gpu_streams = [cp.cuda.Stream(non_blocking=True) for _ in range(3)]
        print("[GPU INFO] Initialized CUDA streams for async operations")

def cleanup_gpu_resources():
    """Clean up GPU resources."""
    global _gpu_streams, _pinned_buffers
    if USE_GPU:
        _gpu_streams = None
        _pinned_buffers.clear()
        cp._default_memory_pool.free_all_blocks()
        cp._default_pinned_memory_pool.free_all_blocks()


def nanpercentile_cupy(arr: cp.ndarray, q: float, axis: int = 0, return_indices: bool = False):
    """
    Compute percentile using GPU, optimized for performance.
    
    Args:
        arr: Input array
        q: Percentile to compute (0-100)
        axis: Axis along which to compute percentile
        return_indices: If True, returns both values and indices
        
    Returns:
        If return_indices is False: percentile values
        If return_indices is True: tuple of (percentile values, indices)
    """
    if arr.ndim < 1:
        raise ValueError("Input array must have at least 1 dimension")

    # Create argsort indices array to track original positions
    indices = None
    if return_indices:
        # Create an index array along the specified axis
        idx_shape = list(arr.shape)
        idx_shape[axis] = 1
        idx = cp.arange(arr.shape[axis], dtype=cp.int64).reshape(
            [1] * axis + [arr.shape[axis]] + [1] * (arr.ndim - axis - 1)
        )
        idx = cp.broadcast_to(idx, arr.shape)
    
    # Try to use native implementation if available (CuPy 12.0+)
    if hasattr(cp, 'nanpercentile') and not return_indices:
        pct = cp.nanpercentile(arr, q, axis=axis)
        return pct.astype(cp.float32)

    # Handle special cases
    if q == 0 and not return_indices:
        return cp.nanmin(arr, axis=axis).astype(cp.float32)
    elif q == 100 and not return_indices:
        return cp.nanmax(arr, axis=axis).astype(cp.float32)

    # Count non-NaN elements along axis
    mask = ~cp.isnan(arr)
    counts = cp.sum(mask, axis=axis)

    # For tracking indices, we need to sort differently
    if return_indices:
        try:
            # Create a sorted indices array that corresponds to sorted values
            nan_mask = cp.isnan(arr)
            
            # Memory-efficient approach: process in slices if data is large
            if arr.size > 100_000_000:  # ~100 million elements threshold
                # Free memory before large operations
                cp._default_memory_pool.free_all_blocks()
                
                # Process in horizontal slices (process rows/columns in batches)
                slice_size = min(1000, arr.shape[1] // 2)  # Process reasonable sized slices
                sorted_arr = cp.empty_like(arr)
                sorted_idx = cp.empty_like(idx)
                
                for start_idx in range(0, arr.shape[1], slice_size):
                    end_idx = min(start_idx + slice_size, arr.shape[1])
                    slice_arr = arr[:, start_idx:end_idx]
                    slice_idx = idx[:, start_idx:end_idx]
                    slice_mask = nan_mask[:, start_idx:end_idx]
                    
                    # Replace NaNs with high value just for this slice
                    temp_slice = cp.where(slice_mask, cp.finfo(arr.dtype).max, slice_arr)
                    # Sort this slice
                    slice_sort_indices = cp.argsort(temp_slice, axis=axis)
                    
                    # Apply sorting to both data and indices
                    sorted_arr[:, start_idx:end_idx] = cp.take_along_axis(
                        slice_arr, slice_sort_indices, axis=axis)
                    sorted_idx[:, start_idx:end_idx] = cp.take_along_axis(
                        slice_idx, slice_sort_indices, axis=axis)
                    
                    # Clean up
                    del temp_slice, slice_sort_indices, slice_arr, slice_idx, slice_mask
                    cp._default_memory_pool.free_all_blocks()
            else:
                # Standard approach for smaller arrays
                # Replace NaNs with a high value for sorting
                temp_arr = cp.where(nan_mask, cp.finfo(arr.dtype).max, arr)
                # Get indices that would sort the array
                sort_indices = cp.argsort(temp_arr, axis=axis)
                # Sort the array using these indices
                sorted_arr = cp.take_along_axis(arr, sort_indices, axis=axis)
                # Sort the index tracker using these same indices
                sorted_idx = cp.take_along_axis(idx, sort_indices, axis=axis)
                
                # Clean up
                del temp_arr, sort_indices
        
        except cp.cuda.memory.OutOfMemoryError:
            print("[GPU INFO] Out of memory during sorting. Falling back to CPU.")
            # Return signal to caller that we need to fall back to CPU
            if return_indices:
                return None, None
            else:
                return None
    else:
        # Simple sorting without index tracking
        try:
            sorted_arr = cp.sort(arr, axis=axis)
        except cp.cuda.memory.OutOfMemoryError:
            print("[GPU INFO] Out of memory during sorting. Falling back to CPU.")
            return None

    # Calculate index position for the percentile
    q_index = (q / 100.0) * (counts - 1)
    q_index_floor = cp.floor(q_index).astype(cp.int32)
    q_index_ceil = cp.ceil(q_index).astype(cp.int32)

    # Handle edge case where all values are NaN
    all_nan_mask = counts == 0

    # Clamp indices to valid range
    q_index_floor = cp.clip(q_index_floor, 0, cp.maximum(counts - 1, 0))
    q_index_ceil = cp.clip(q_index_ceil, 0, cp.maximum(counts - 1, 0))

    # Select values and interpolate
    if arr.ndim > 1:
        floor_vals = cp.take_along_axis(sorted_arr,
                                        cp.expand_dims(q_index_floor, axis=axis),
                                        axis=axis).squeeze(axis=axis)
        ceil_vals = cp.take_along_axis(sorted_arr,
                                       cp.expand_dims(q_index_ceil, axis=axis),
                                       axis=axis).squeeze(axis=axis)
        
        if return_indices:
            # Also extract the indices that gave these values
            floor_indices = cp.take_along_axis(sorted_idx,
                                             cp.expand_dims(q_index_floor, axis=axis),
                                             axis=axis).squeeze(axis=axis)
    else:
        floor_vals = sorted_arr[q_index_floor]
        ceil_vals = sorted_arr[q_index_ceil]
        
        if return_indices:
            floor_indices = sorted_idx[q_index_floor]

    # Linear interpolation
    frac = q_index - q_index_floor
    result = floor_vals + (ceil_vals - floor_vals) * frac

    # Set result to NaN where all inputs were NaN
    if all_nan_mask.any():
        result[all_nan_mask] = cp.nan
        if return_indices:
            # Mark invalid indices with -1
            floor_indices = floor_indices.copy()  # Ensure we don't modify the original
            floor_indices[all_nan_mask] = -1

    if return_indices:
        return result.astype(cp.float32), floor_indices.astype(cp.int64)
    else:
        return result.astype(cp.float32)


def process_chunk_gpu(scene_chunks):
    """
    Process a window using GPU (CuPy) for high-performance percentile compositing.
    
    OPTIMIZATIONS APPLIED:
    1. Batch CPU->GPU transfer: Stack all data on CPU first, single transfer
    2. Pinned memory: Use page-locked memory for faster transfers
    3. Reduced synchronization: Only sync when absolutely necessary
    4. Pre-allocated INVALID_SCL lookup table on GPU
    
    Returns:
        RGBN values array (4 bands)
    """
    if not scene_chunks:
        return None, None

    # Get dimensions and prepare output
    n_scenes = len(scene_chunks)
    c, h, w = scene_chunks[0]["RGB"].shape  # c==4
    nodata_val = RGB_NODATA
    
    # Check if memory is likely to be sufficient - if not, switch to CPU immediately
    try:
        device = cp.cuda.Device(0)
        mem_free_start = device.mem_info[0] / (1024**3)
        
        # Estimate memory requirements (very rough estimation)
        # 4 bytes per float32 element, 8 bytes per int64 element
        approx_mem_needed = n_scenes * 4 * h * w * 4 * 4 / (1024**3)  # in GB
        
        if approx_mem_needed > mem_free_start * 0.8:
            print(f"[GPU INFO] Estimated memory needed ({approx_mem_needed:.2f}GB) exceeds available memory ({mem_free_start:.2f}GB)")
            print("[GPU INFO] Preemptively switching to CPU processing")
            return process_chunk_cpu(scene_chunks)
        
        # =====================================================================
        # OPTIMIZATION 1: Batch data preparation on CPU (avoid many small transfers)
        # =====================================================================
        
        # Pre-allocate CPU arrays for batch stacking
        cpu_rgbn_stack = np.empty((n_scenes, 4, h, w), dtype=np.float32)
        cpu_scl_stack = np.empty((n_scenes, h, w), dtype=np.uint8)
        cpu_cloud_stack = np.empty((n_scenes, h, w), dtype=np.uint8)
        cpu_day_indices = np.empty(n_scenes, dtype=np.uint32)
        
        # Stack all scene data on CPU (fast NumPy operations)
        for i, sc in enumerate(scene_chunks):
            cpu_rgbn_stack[i] = sc["RGB"].astype(np.float32)
            cpu_scl_stack[i] = sc["SCL"]
            cpu_cloud_stack[i] = sc["OmniCloudMask"]
            cpu_day_indices[i] = sc.get("day_idx", 0)
        
        # =====================================================================
        # OPTIMIZATION 2: Single bulk transfer to GPU using pinned memory
        # =====================================================================
        
        # Use pinned memory for faster CPU->GPU transfer
        try:
            # Allocate pinned memory and copy data
            pinned_rgbn = cp.cuda.alloc_pinned_memory(cpu_rgbn_stack.nbytes)
            pinned_rgbn_array = np.frombuffer(pinned_rgbn, dtype=np.float32).reshape(cpu_rgbn_stack.shape)
            np.copyto(pinned_rgbn_array, cpu_rgbn_stack)
            
            # Async transfer using default stream (will be faster with pinned memory)
            all_rgbn = cp.asarray(pinned_rgbn_array)
        except Exception:
            # Fallback to regular transfer if pinned memory fails
            all_rgbn = cp.asarray(cpu_rgbn_stack)
        
        # Transfer masks in bulk
        all_scl = cp.asarray(cpu_scl_stack)
        all_cloud = cp.asarray(cpu_cloud_stack)
        day_indices_gpu = cp.asarray(cpu_day_indices)
        
        # Free CPU memory early
        del cpu_rgbn_stack, cpu_scl_stack, cpu_cloud_stack
        
        # =====================================================================
        # OPTIMIZATION 3: Vectorized mask computation (no Python loop)
        # =====================================================================
        
        # Pre-compute invalid SCL lookup table on GPU (compute once, reuse)
        scl_invalid_lookup = cp.zeros(256, dtype=cp.bool_)
        for scl_val in INVALID_SCL_CLASSES:
            if scl_val < 256:
                scl_invalid_lookup[scl_val] = True
        
        # Vectorized SCL mask lookup for all scenes at once
        all_scl_invalid = scl_invalid_lookup[all_scl]
        
        # Vectorized nodata detection
        all_rgbn_invalid = cp.any(all_rgbn == float(nodata_val), axis=1)
        
        # Vectorized cloud mask
        all_cloud_invalid = all_cloud > 0
        
        # Combine all masks (still fully vectorized)
        all_invalid = all_scl_invalid | all_rgbn_invalid | all_cloud_invalid
        
        # Free intermediate mask arrays
        del all_scl, all_cloud, all_scl_invalid, all_rgbn_invalid, all_cloud_invalid, scl_invalid_lookup
        
        # =====================================================================
        # OPTIMIZATION 4: Apply mask and compute percentile
        # =====================================================================
        
        # Apply mask using broadcasting
        invalid_mask_4d = cp.expand_dims(all_invalid, axis=1)
        rgbn_masked = cp.where(cp.broadcast_to(invalid_mask_4d, all_rgbn.shape), cp.nan, all_rgbn)
        
        # Free original arrays
        del all_rgbn, all_invalid, invalid_mask_4d
        
        # =====================================================================
        # OPTIMIZATION 5: Compute percentile for all bands (batched if possible)
        # =====================================================================
        
        output_bands = cp.empty((4, h, w), dtype=cp.float32)
        
        # Process all 4 bands
        for b in range(4):
            band_data = rgbn_masked[:, b]
            band_values, band_indices = nanpercentile_cupy(
                band_data, float(PERCENTILE_VALUE), axis=0, return_indices=True)
            
            if band_values is None:
                # OOM during percentile - fall back to CPU
                del rgbn_masked
                cp._default_memory_pool.free_all_blocks()
                return process_chunk_cpu(scene_chunks)
            
            output_bands[b] = band_values
            
            del band_values, band_indices
        
        # Replace NaN with nodata
        output_bands = cp.nan_to_num(output_bands, nan=float(nodata_val))
        
        # Clean up
        del rgbn_masked
        
        # =====================================================================
        # OPTIMIZATION 7: Single sync before transfer back
        # =====================================================================
        
        # Only sync once before transferring results
        cp.cuda.Device().synchronize()
        
        # Transfer results back to CPU
        out_rgb = cp.asnumpy(output_bands).astype(RGB_DTYPE)
        
        # Cleanup GPU memory
        del output_bands
        cp._default_memory_pool.free_all_blocks()

    except Exception as e:
        print(f"[WARNING] GPU processing failed: {e}")
        print("[WARNING] Falling back to CPU for this chunk")
        
        # Fall back to CPU processing for this chunk
        out_rgb = process_chunk_cpu(scene_chunks)
        
        # Clear GPU memory after error
        cp._default_memory_pool.free_all_blocks()
        cp._default_pinned_memory_pool.free_all_blocks()

    return out_rgb

###############################################################################
# CPU implementation (fallback for systems without GPU)
###############################################################################


def process_chunk_cpu(scene_chunks):
    """
    Process a chunk with CPU (fallback if GPU unavailable or fails).
    
    Returns:
        RGBN values array (4 bands)
    """
    if not scene_chunks:
        return None, None

    n_scenes = len(scene_chunks)
    c, h, w = scene_chunks[0]["RGB"].shape  # c==4
    nodata_val = RGB_NODATA

    # Create stacks for all bands and masks
    rgbn_stack = np.empty((n_scenes, 4, h, w), dtype=np.float32)
    invalid_mask = np.empty((n_scenes, h, w), dtype=bool)

    for i, sc in enumerate(scene_chunks):
        # Extract data
        rgbn_stack[i] = sc["RGB"]
        sub_scl = sc["SCL"]
        sub_cloud = sc["OmniCloudMask"]

        # Build invalid mask
        scl_invalid = np.isin(sub_scl, list(INVALID_SCL_CLASSES))
        rgbn_invalid = np.any(rgbn_stack[i] == float(nodata_val), axis=0)
        cloud_invalid = sub_cloud > 0

        invalid_mask[i] = scl_invalid | rgbn_invalid | cloud_invalid

    # Allocate output arrays
    out_rgbn = np.full((4, h, w), nodata_val, dtype=RGB_DTYPE)

    # Process each band
    for b in range(4):
        band = rgbn_stack[:, b, :, :].copy()
        
        # Get percentile values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            pct_vals = np.nanpercentile(band, PERCENTILE_VALUE, axis=0)
        
        # Replace NaNs with nodata
        pct_vals = np.nan_to_num(pct_vals, nan=float(nodata_val))
        out_rgbn[b] = pct_vals.astype(RGB_DTYPE)

    return out_rgbn

###############################################################################
# File and metadata handling functions
###############################################################################


def parse_s2_filename(fname):
    """
    Parse Sentinel-2 filename to extract tile, date and day index.
    E.g. S2B_15SUA_20240121_0_L2A_B04.tif -> (tile="15SUA", date_str="20240121", day_idx=0)
    """
    m = FILENAME_REGEX.search(fname)
    if not m:
        return None, None, None
    tile = m.group(1)
    date_str = m.group(2)
    day_idx = int(m.group(3))
    return (tile, date_str, day_idx)


def read_scene_chunk(scene_info, win_bounds, chunk_shape):
    """
    Read and reproject a portion of a scene that overlaps with the given window.
    Returns a dict with RGB, SCL, OmniCloudMask arrays and day_idx, or None if no overlap.
    """
    band_files = scene_info["bands"]
    mosaic_bounds = scene_info["mosaic_bounds"]
    day_idx = scene_info.get("day_idx", 0)  # Extract day index or default to 0

    # Quick boundary check
    left = max(win_bounds[0], mosaic_bounds[0])
    right = min(win_bounds[2], mosaic_bounds[2])
    bottom = max(win_bounds[1], mosaic_bounds[1])
    top = min(win_bounds[3], mosaic_bounds[3])

    if (left >= right) or (bottom >= top):
        return None  # No overlap

    chunk_h, chunk_w = chunk_shape
    if chunk_h <= 0 or chunk_w <= 0:
        return None

    # Build transform
    chunk_affine = Affine(
        TARGET_RES_X, 0.0, win_bounds[0],
        0.0, -TARGET_RES_Y, win_bounds[3]
    )

    nodata_val = RGB_NODATA

    # Parallel band reading function
    def read_band(band_key, output_type, output_nodata, band_idx=None):
        if band_key not in band_files:
            return None

        output_array = np.full(
            (chunk_h, chunk_w), output_nodata, dtype=output_type)

        with rasterio.open(band_files[band_key]) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=output_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=chunk_affine,
                dst_crs=TARGET_CRS,
                resampling=Resampling.nearest,
                dst_nodata=output_nodata
            )

        return (band_idx, output_array) if band_idx is not None else output_array

    # Pre-allocate arrays
    rgbn_arr = np.full((4, chunk_h, chunk_w), nodata_val, dtype=RGB_DTYPE)
    scl_arr = np.full((chunk_h, chunk_w), 0, dtype=np.uint8)
    cloud_arr = np.full((chunk_h, chunk_w), 0, dtype=np.uint8)

    # Define band reading tasks
    band_tasks = [
        ("B04", RGB_DTYPE, nodata_val, 0),  # R
        ("B03", RGB_DTYPE, nodata_val, 1),  # G
        ("B02", RGB_DTYPE, nodata_val, 2),  # B
        ("B08", RGB_DTYPE, nodata_val, 3),  # NIR
        ("SCL", np.uint8, 0, None),
        ("OmniCloudMask", np.uint8, 0, None)
    ]

    # Run band reading in parallel (optimized thread count)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for band_key, dtype, nodata, band_idx in band_tasks:
            futures.append(executor.submit(
                read_band, band_key, dtype, nodata, band_idx))

        # Process results
        for i, future in enumerate(futures):
            result = future.result()
            if result is None:
                continue

            if i < 4:  # RGBN bands
                band_idx, band_data = result
                rgbn_arr[band_idx] = band_data
            elif i == 4:  # SCL
                scl_arr = result
            elif i == 5:  # OmniCloudMask
                cloud_arr = result

    return {
        "RGB": rgbn_arr,
        "SCL": scl_arr,
        "OmniCloudMask": cloud_arr
    }

###############################################################################
# Main processing function
###############################################################################


def main():
    print("\n=== Sentinel-2 Monthly Mosaic Generator v22 (GPU-optimized, Sequential) ===\n")

    ############################################################################
    # 1) Load configuration
    ############################################################################
    cfg_file = "config.json"
    if not os.path.exists(cfg_file):
        print("[ERROR] config.json not found, exiting.")
        sys.exit(1)

    # Load metadata if available
    if os.path.exists("metadata.json"):
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    with open(cfg_file, "r", encoding="utf-8") as cf:
        config = json.load(cf)

    # Get user parameters
    global PERCENTILE_VALUE, TARGET_CRS
    PERCENTILE_VALUE = float(config.get("percentileValue", 25.0))
    
    # Load target CRS from config (required)
    TARGET_CRS = config.get("target_crs")
    if not TARGET_CRS:
        print("[ERROR] 'target_crs' not found in config.json. Please specify the target CRS (e.g., 'EPSG:32615').")
        sys.exit(1)
    print(f"[INFO] Target CRS: {TARGET_CRS}")

    print(f"[INFO] Processing mode: {'GPU' if USE_GPU else 'CPU'}")
    print(f"[INFO] Using percentileValue={PERCENTILE_VALUE}")

    # Parse date range and output directory
    start_date_str = config.get("start_date", config.get("start_data", "2025-02-01"))
    end_date_str = config.get("end_date", None)
    
    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        
        # Set end date to end of month if not specified
        if end_date_str:
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        else:
            # Default to end of month if no end date specified
            if start_dt.month == 12:
                end_dt = datetime(start_dt.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_dt = datetime(start_dt.year, start_dt.month + 1, 1) - timedelta(days=1)
        
        # Create date range string for filename
        date_range_str = f"{start_dt.strftime('%Y-%m-%d')}_to_{end_dt.strftime('%Y-%m-%d')}"
        print(f"[INFO] Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    except ValueError as e:
        print(f"[WARNING] Date parsing error: {e}")
        date_range_str = "YYYY-MM-DD_to_YYYY-MM-DD"
        start_dt = None
        end_dt = None

    data_dir = config.get("cleaned_dir", None)
    if not os.path.isdir(data_dir):
        print("[ERROR] data_dir not found.")
        sys.exit(1)

    # Check for user-supplied coverage
    user_cov = config.get("bbox")
    user_coverage_gdf = None
    if user_cov and os.path.exists(user_cov):
        print(f"Reading bbox file: {user_cov}")
        
        # Validate file format
        supported_formats = ['.gpkg', '.shp', '.geojson', '.json']
        file_ext = os.path.splitext(user_cov)[1].lower()
        
        if file_ext not in supported_formats:
            print(f"Warning: File extension '{file_ext}' may not be supported.")
            print(f"Supported formats: {', '.join(supported_formats)}")
        else:
            print(f"Detected bbox format: {file_ext}")
        
        try:
            user_coverage_gdf = gpd.read_file(user_cov)
        except Exception as e:
            print(f"Error reading bbox file '{user_cov}': {e}")
            print("Ensure the file is a valid vector format (gpkg, shp, geojson)")
            sys.exit(1)
        
        if user_coverage_gdf.crs is None:
            raise ValueError(f"[ERROR] {user_cov} has no CRS.")
        
        print(f"Bbox CRS: {user_coverage_gdf.crs}")
        
        if user_coverage_gdf.crs.to_string() != TARGET_CRS:
            print(f"[INFO] Reprojecting coverage -> {TARGET_CRS}...")
            user_coverage_gdf = user_coverage_gdf.to_crs(TARGET_CRS)
            tmp_cov = f"temp_coverage_{TARGET_CRS.replace(':', '_')}.gpkg"
            user_coverage_gdf.to_file(tmp_cov, driver="GPKG")
            print(f"[INFO] Wrote {tmp_cov}")
    else:
        if user_cov:
            print(
                f"[WARNING] Coverage file={user_cov} not found, ignoring coverage.")

    ############################################################################
    # 2) Gather S2 L2A files for B02, B03, B04, plus SCL + OmniCloudMask
    ############################################################################
    print("[INFO] Scanning for Sentinel-2 files...")
    relevant_bands = ["B02.tif", "B03.tif", "B04.tif",
                      "B08.tif", "SCL.tif", "OmniCloudMask.tif"]
    scenes_dict = {}

    for fname in os.listdir(data_dir):
        if not any(fname.endswith(rb) for rb in relevant_bands):
            continue

        tile, date_str, day_idx = parse_s2_filename(fname)
        if tile is None or date_str is None:
            continue

        full_path = os.path.join(data_dir, fname)
        # Identify band key
        if fname.endswith("B02.tif"):
            band_key = "B02"
        elif fname.endswith("B03.tif"):
            band_key = "B03"
        elif fname.endswith("B04.tif"):
            band_key = "B04"
        elif fname.endswith("B08.tif"):
            band_key = "B08"
        elif fname.endswith("SCL.tif"):
            band_key = "SCL"
        elif fname.endswith("OmniCloudMask.tif"):
            band_key = "OmniCloudMask"
        else:
            continue

        key = (tile, date_str)
        if key not in scenes_dict:
            scenes_dict[key] = {}
        if day_idx not in scenes_dict[key]:
            scenes_dict[key][day_idx] = {"bands": {}}

        scenes_dict[key][day_idx]["bands"][band_key] = full_path

    # Keep scenes with different day indices, filtered by date range
    final_scenes = []
    filtered_scenes = 0
    print("[INFO] Scene day indices found in files:")
    
    for (tile, dstr), day_map in scenes_dict.items():
        # Use the actual day of month from the date instead of the day_idx from filename
        for day_idx, day_info in day_map.items():
            band_dict = day_info["bands"]
            needed = {"B02", "B03", "B04", "B08", "SCL", "OmniCloudMask"}
            if needed.issubset(band_dict.keys()):
                date_obj = datetime.strptime(dstr, "%Y%m%d")
                
                # Filter by date range if valid date ranges were parsed
                if start_dt and end_dt:
                    # Skip scenes outside the specified date range
                    if date_obj < start_dt or date_obj > end_dt:
                        filtered_scenes += 1
                        continue
                
                print(f"  Tile: {tile}, Date: {dstr}, Day: {day_idx}")
                final_scenes.append({
                    "date_obj": date_obj,
                    "tile": tile,
                    "date_str": dstr,
                    "bands": band_dict
                })
    
    if filtered_scenes > 0:
        print(f"[INFO] Filtered out {filtered_scenes} scenes outside the date range.")

    if not final_scenes:
        print("[ERROR] No scenes found with B02/B03/B04/SCL/OmniCloudMask.")
        sys.exit(0)

    final_scenes.sort(key=lambda x: x["date_obj"])
    print(f"[INFO] Found {len(final_scenes)} scenes with required bands.")

    # Detect data type and set nodata
    global RGB_DTYPE, RGB_NODATA
    sample_b4_path = final_scenes[0]["bands"]["B04"]
    with rasterio.open(sample_b4_path) as sample_src:
        RGB_DTYPE = sample_src.profile["dtype"]
        print(f"[INFO] Detected band dtype = {RGB_DTYPE}")

    # Set appropriate nodata value
    if "int" in RGB_DTYPE:
        RGB_NODATA = 0  # Sentinel L2A reflectance is often uint16, with 0 as no-data
    else:
        RGB_NODATA = 0  # For float types

    ############################################################################
    # 3) Build coverage from scenes + optional user coverage
    ############################################################################
    print("[INFO] Computing coverage...")
    coverage_polygons = []
    scene_infos = []

    for item in final_scenes:
        band_dict = item["bands"]
        sample_b4 = band_dict["B04"]
        with rasterio.open(sample_b4) as src:
            src_crs = src.crs
            src_bounds = src.bounds

        mosaic_bounds = transform_bounds(
            src_crs, TARGET_CRS,
            src_bounds.left, src_bounds.bottom,
            src_bounds.right, src_bounds.top,
            densify_pts=21
        )
        coverage_polygons.append(box(*mosaic_bounds))

        scene_infos.append({
            "bands": band_dict,
            "src_crs": src_crs,
            "src_bounds": src_bounds,
            "mosaic_bounds": mosaic_bounds
        })

    if user_coverage_gdf is not None:
        coverage_polygons.extend(list(user_coverage_gdf.geometry))

    coverage_gdf = gpd.GeoDataFrame(geometry=coverage_polygons, crs=TARGET_CRS)
    coverage_gdf.to_file("s2_coverage.gpkg", driver="GPKG")
    print(f"[INFO] Wrote s2_coverage.gpkg in {TARGET_CRS}.")

    # Compute overall mosaic bounds with validation
    print("[INFO] Validating polygon bounds...")
    valid_polygons = []

    for i, p in enumerate(coverage_polygons):
        bounds = p.bounds
        # Check for invalid bounds (infinity, NaN)
        if (not any(math.isinf(b) for b in bounds) and
            not any(math.isnan(b) for b in bounds) and
                bounds[0] < bounds[2] and bounds[1] < bounds[3]):
            valid_polygons.append(p)
        else:
            print(
                f"[WARNING] Skipping polygon {i} with invalid bounds: {bounds}")

    if not valid_polygons:
        print("[ERROR] No valid polygons found. Check your input data.")
        sys.exit(1)

    print(
        f"[INFO] Using {len(valid_polygons)} valid polygons out of {len(coverage_polygons)} total")

    mosaic_left = min(p.bounds[0] for p in valid_polygons)
    mosaic_bottom = min(p.bounds[1] for p in valid_polygons)
    mosaic_right = max(p.bounds[2] for p in valid_polygons)
    mosaic_top = max(p.bounds[3] for p in valid_polygons)

    print(
        f"[INFO] Mosaic bounds: ({mosaic_left}, {mosaic_bottom}) - ({mosaic_right}, {mosaic_top})")

    mosaic_width = int(math.ceil((mosaic_right - mosaic_left) / TARGET_RES_X))
    mosaic_height = int(math.ceil((mosaic_top - mosaic_bottom) / TARGET_RES_Y))
    mosaic_transform = Affine(
        TARGET_RES_X, 0.0, mosaic_left,
        0.0, -TARGET_RES_Y, mosaic_top
    )

    ############################################################################
    # 4) Prepare output mosaic file
    ############################################################################
    out_profile = {
        "driver": "GTiff",
        "dtype": RGB_DTYPE,
        "count": 4,  # R,G,B,NIR
        "width": mosaic_width,
        "height": mosaic_height,
        "crs": TARGET_CRS,
        "transform": mosaic_transform,
        "nodata": RGB_NODATA,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "YES"
    }

    output_dir = config.get("output_dir", ".")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get configurable output filename base from config
    output_filename_base = config.get("output_filename_base", "Arkansas_S2_RGBN")

    # Define output filename using date range
    mode_suffix = "_gpu" if USE_GPU else "_cpu"
    mosaic_name = os.path.join(
        output_dir, f"{output_filename_base}_{date_range_str}.tif")

    # Create output file
    with rasterio.open(mosaic_name, "w", **out_profile) as dst:
        pass
    print(f"[INFO] Created empty 4-band RGBN mosaic: {mosaic_name}")

    ############################################################################
    # 5) Process chunks
    ############################################################################
    # Use GPU chunk size
    CHUNK_SIZE = GPU_CHUNK_SIZE
    
    # Initialize GPU resources for optimal performance
    if USE_GPU:
        init_gpu_resources()

    # Create windows for processing
    windows = []
    for row_off in range(0, mosaic_height, CHUNK_SIZE):
        for col_off in range(0, mosaic_width, CHUNK_SIZE):
            chunk_width = min(CHUNK_SIZE, mosaic_width - col_off)
            chunk_height = min(CHUNK_SIZE, mosaic_height - row_off)
            chunk_win = Window(col_off=col_off, row_off=row_off,
                               width=chunk_width, height=chunk_height)
            windows.append(chunk_win)

    print(
        f"[INFO] Processing {len(windows)} chunks with PIPELINED I/O, size {CHUNK_SIZE}x{CHUNK_SIZE}")

    # Helper function to compute window bounds
    def get_window_info(chunk_win):
        """Compute bounds and shape for a window."""
        row_off = chunk_win.row_off
        col_off = chunk_win.col_off
        w = chunk_win.width
        h = chunk_win.height
        
        x_ul, y_ul = rasterio.transform.xy(
            mosaic_transform, row_off, col_off, offset="ul"
        )
        x_lr, y_lr = rasterio.transform.xy(
            mosaic_transform, row_off + h - 1, col_off + w - 1, offset="lr"
        )
        
        left = min(x_ul, x_lr)
        right = max(x_ul, x_lr)
        top = max(y_ul, y_lr)
        bottom = min(y_ul, y_lr)
        
        return (left, bottom, right, top), (h, w)
    
    # Helper function to read all scenes for a chunk (for async prefetch)
    def prefetch_chunk_data(chunk_win, scene_infos_list):
        """Read scene data for a chunk - runs in thread pool for prefetching."""
        win_bounds, chunk_shape = get_window_info(chunk_win)
        scene_data = []
        for si in scene_infos_list:
            sc = read_scene_chunk(si, win_bounds, chunk_shape)
            if sc is not None:
                scene_data.append(sc)
        return scene_data, chunk_win
    
    # Process chunks in parallel for maximum GPU utilization
    start_time = time.time()
    total_chunks = len(windows)
    
    # Determine optimal number of parallel workers based on GPU memory
    if USE_GPU:
        device = cp.cuda.Device(0)
        mem_total_gb = device.mem_info[1] / (1024**3)
        # Process 2-4 chunks in parallel depending on GPU memory
        if mem_total_gb > 20:
            num_workers = 8  # Large GPU - process 4 chunks in parallel
        elif mem_total_gb > 10:
            num_workers = 3  # Medium GPU - process 3 chunks in parallel
        else:
            num_workers = 2  # Small GPU - process 2 chunks in parallel
    else:
        # CPU mode: use more workers since no GPU memory constraint
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count() // 2, 8)
    
    print(f"[INFO] Using {num_workers} parallel workers for chunk processing")
    
    # Function to process a single chunk (for parallel execution)
    def process_single_chunk(chunk_idx):
        """Process a single chunk and return results with window info."""
        chunk_win = windows[chunk_idx]
        h = chunk_win.height
        w = chunk_win.width
        
        # Read scene data for this chunk
        win_bounds, chunk_shape = get_window_info(chunk_win)
        scene_chunks = []
        for si in scene_infos:
            sc = read_scene_chunk(si, win_bounds, chunk_shape)
            if sc is not None:
                scene_chunks.append(sc)
        
        if not scene_chunks:
            # No overlap, return nodata
            out_r = np.full((h, w), RGB_NODATA, dtype=RGB_DTYPE)
            out_g = np.full((h, w), RGB_NODATA, dtype=RGB_DTYPE)
            out_b = np.full((h, w), RGB_NODATA, dtype=RGB_DTYPE)
            out_nir = np.full((h, w), RGB_NODATA, dtype=RGB_DTYPE)
            return (chunk_win, out_r, out_g, out_b, out_nir)
        
        # Process chunk with GPU if available, CPU otherwise
        if USE_GPU:
            rgb_data = process_chunk_gpu(scene_chunks)
        else:
            rgb_data = process_chunk_cpu(scene_chunks)
        
        if rgb_data is None:
            # All invalid, return nodata
            out_r = np.full((h, w), RGB_NODATA, dtype=RGB_DTYPE)
            out_g = np.full((h, w), RGB_NODATA, dtype=RGB_DTYPE)
            out_b = np.full((h, w), RGB_NODATA, dtype=RGB_DTYPE)
            out_nir = np.full((h, w), RGB_NODATA, dtype=RGB_DTYPE)
        else:
            # Unpack the results
            out_r, out_g, out_b, out_nir = rgb_data[0], rgb_data[1], rgb_data[2], rgb_data[3]
        
        return (chunk_win, out_r, out_g, out_b, out_nir)
    
    # Create executor for parallel chunk processing
    from concurrent.futures import ThreadPoolExecutor as ParallelExecutor
    
    # Open the main mosaic file for writing
    with rasterio.open(mosaic_name, "r+") as dst_rgb:
        
        # Process chunks in parallel batches
        with ParallelExecutor(max_workers=num_workers) as executor:
            # Submit all chunks for processing
            futures = []
            for i in range(total_chunks):
                future = executor.submit(process_single_chunk, i)
                futures.append((i, future))
            
            # Collect results and write as they complete
            completed = 0
            for i, future in futures:
                result = future.result()
                chunk_win, out_r, out_g, out_b, out_nir = result
                
                # Write to RGB file
                dst_rgb.write(out_r, 1, window=chunk_win)
                dst_rgb.write(out_g, 2, window=chunk_win)
                dst_rgb.write(out_b, 3, window=chunk_win)
                dst_rgb.write(out_nir, 4, window=chunk_win)
                
                # Progress reporting
                completed += 1
                if completed % 10 == 0 or completed == 1:
                    progress = (completed / total_chunks) * 100
                    elapsed = time.time() - start_time
                    per_chunk = elapsed / completed
                    eta = per_chunk * (total_chunks - completed)
                    print(
                        f"[INFO] Processing: {progress:.1f}% ({completed}/{total_chunks}), "
                        f"Speed: {per_chunk:.2f}s/chunk, ETA: {eta:.1f}s")
    
    # Cleanup GPU resources
    if USE_GPU:
        cleanup_gpu_resources()

    print(f"[DONE] Created 4-band RGBN percentile mosaic => {mosaic_name}")

    ############################################################################
    # 6) Create a VRT for convenience
    ############################################################################
    vrt_name = mosaic_name.replace(".tif", ".vrt")
    cmd = f"gdal_translate -of VRT -a_srs {TARGET_CRS} {mosaic_name} {vrt_name}"
    print(f"[INFO] Creating VRT => {vrt_name}")
    os.system(cmd)

    ############################################################################
    # 7) Build overviews for proper COG structure
    ############################################################################
    print("[INFO] Building overviews with levels 4, 8, 16, 32...")
    
    # Build overviews for the main RGBN mosaic
    overview_cmd = f"gdaladdo -r average --config COMPRESS_OVERVIEW DEFLATE --config PREDICTOR_OVERVIEW 2 {mosaic_name} 4 8 16 32"
    print(f"[INFO] Running: {overview_cmd}")
    os.system(overview_cmd)
    print(f"[INFO] Added overviews to {mosaic_name}")

    ############################################################################
    # 8) Create comprehensive metadata.json
    ############################################################################
    print("[INFO] Creating comprehensive metadata.json...")
    
    # Update metadata with all relevant information
    metadata["mosaicFile"] = mosaic_name
    metadata["vrtFile"] = vrt_name
    metadata["processingMode"] = "gpu" if USE_GPU else "cpu"
    metadata["percentileValue"] = PERCENTILE_VALUE
    metadata["targetCRS"] = TARGET_CRS
    metadata["resolution"] = {
        "x": TARGET_RES_X,
        "y": TARGET_RES_Y
    }
    metadata["bounds"] = {
        "left": mosaic_left,
        "bottom": mosaic_bottom,
        "right": mosaic_right,
        "top": mosaic_top
    }
    metadata["dimensions"] = {
        "width": mosaic_width,
        "height": mosaic_height
    }
    metadata["dateRange"] = {
        "start": start_date_str,
        "end": end_date_str if end_date_str else "end of month",
        "rangeString": date_range_str
    }
    metadata["scenes"] = {
        "total": len(final_scenes),
        "firstDate": final_scenes[0]["date_obj"].strftime("%Y-%m-%d"),
        "lastDate": final_scenes[-1]["date_obj"].strftime("%Y-%m-%d")
    }
    metadata["bands"] = ["Red (B04)", "Green (B03)", "Blue (B02)", "NIR (B08)"]
    metadata["bandCount"] = 4
    metadata["dtype"] = RGB_DTYPE
    metadata["nodata"] = float(RGB_NODATA) if isinstance(RGB_NODATA, (int, float)) else RGB_NODATA
    metadata["overviews"] = [4, 8, 16, 32]
    metadata["compression"] = "deflate"
    metadata["tiled"] = True
    metadata["chunkSize"] = CHUNK_SIZE
    metadata["processingDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata["coverageFile"] = "s2_coverage.gpkg"
    
    # Save metadata.json
    with open("metadata.json", "w") as mf:
        json.dump(metadata, mf, indent=2)
    
    print("[INFO] Metadata saved to metadata.json")
    print(f"[INFO] Metadata includes: mosaic file, bounds, CRS, date range, {len(final_scenes)} scenes")

    # Print final stats
    total_time = time.time() - start_time
    print(
        f"[INFO] Processing completed in {'GPU' if USE_GPU else 'CPU'} mode.")
    print(f"[INFO] Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
