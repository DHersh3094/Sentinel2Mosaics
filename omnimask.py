#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
createOmniCloudMasks.py

Iterates over each Sentinel-2 scene in a "cleaned_data/" directory.
- For each unique (mission, tile, date_str, day_idx), tries to gather
  the 3 desired bands: B04 (red), B03 (green), B08 (nir).
- If any are missing, that scene is skipped.
- Each .tif is opened at its native resolution (no reprojection).
- We optionally move each band to GPU (if cupy is available and USE_GPU=True)
  for reflectance scaling and nodata replacement, then stack into a
  (3, height, width) array of reflectances in [0..1].
- We pass that array to OmniCloudMask (predict_from_array), returning a
  single-band mask with these codes:
    0 = Clear
    1 = Thick Cloud
    2 = Thin Cloud
    3 = Shadow
- If GPU inference fails with OOM, we disable GPU and retry on CPU.
- We then write the final classification mask as a single-band GeoTIFF:
     ..._OmniCloudMask.tif
  with nodata=255 for pixels that were nodata in any input band.

Parallelization:
- Scenes are processed in parallel using ProcessPoolExecutor (spawn mode).
- Each scene is handled in its own process. If you have limited GPU memory, you
  may wish to reduce the number of parallel workers (e.g. max_workers=1 or 2).

Environment:
- Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True can sometimes help
  with GPU memory fragmentation.
"""

import os
import sys
import json
import re
import numpy as np
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed

# OmniCloudMask
from omnicloudmask import predict_from_array

# ukis-pysat for writing output GeoTIFFs (you can replace with direct rasterio if preferred).
from ukis_pysat.raster import Image

###############################################################################
# (Optional) Environment variable to reduce GPU memory fragmentation
###############################################################################
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

###############################################################################
# Try to import cupy. If not available, fall back to CPU-only numpy.
###############################################################################
USE_GPU = True
try:
    import cupy as cp
except ImportError:
    print("[WARNING] CuPy not available. Using CPU-only numpy for reflectance scaling.")
    USE_GPU = False

###############################################################################
# Sentinel-2 bands -> OmniCloudMask typically needs Red/Green/NIR in that order
###############################################################################
BAND_MAPPING = {
    "B04": "red",    # ~665 nm
    "B03": "green",  # ~560 nm
    "B08": "nir",    # ~842 nm
}

RELEVANT_BAND_FILES = [b + ".tif" for b in BAND_MAPPING.keys()]

###############################################################################
# Regex to parse Sentinel-2 L2A filenames, e.g. S2A_15SVA_20250201_0_L2A_B04.tif
###############################################################################
FILENAME_REGEX = re.compile(r"(S2[ABC])_(\d{2}[A-Z]{3})_(\d{8})_(\d+)_L2A_")

def parse_s2_filename(fname):
    """
    Example: "S2A_15SVA_20250201_0_L2A_B04.tif"
        -> (mission='S2A', tile='15SVA', date_str='20250201', day_idx=0)
    Returns (None, None, None, None) if not matched.
    """
    m = FILENAME_REGEX.search(fname)
    if not m:
        return None, None, None, None
    mission = m.group(1)  # e.g. "S2A"
    tile = m.group(2)     # e.g. "15SVA"
    date_str = m.group(3) # e.g. "20250201"
    day_idx = int(m.group(4))
    return (mission, tile, date_str, day_idx)

def run_omnimask_on_cpu(stack_cpu):
    """
    Run OmniCloudMask on CPU with batch_size=1, patch_size=393, patch_overlap=196.
    """
    print("   [OMNI] Retrying on CPU with batch_size=1 (forced CPU).")
    pred_mask = predict_from_array(
        stack_cpu,
        batch_size=1,
        mosaic_device="cpu",
        patch_size=393,
        patch_overlap=196
    )
    return pred_mask

def run_omnimask_on_gpu(stack_cpu):
    """
    Run OmniCloudMask on GPU with small batch_size=96 (you can change),
    half precision, and limited patch_size/overlap to avoid large memory usage.
    """
    print("   [OMNI] Attempting GPU with batch_size=96, inference_dtype='float16'...")
    pred_mask = predict_from_array(
        stack_cpu,
        batch_size=96,
        inference_dtype="float16",
        mosaic_device="cuda",
        patch_size=393,
        patch_overlap=196
    )
    return pred_mask

def disable_gpu_globally():
    """
    Disable GPU for this process by hiding CUDA devices.
    Useful if repeated GPU OOM occurs.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("   [INFO] Disabled GPU visibility for this process.")

def process_scene(key, band_dict, data_dir):
    """
    Process a single scene:
      1) Build the output filename => ..._OmniCloudMask.tif
         - If it already exists, skip processing
      2) Open each band (B04=red, B03=green, B08=nir).
      3) Identify any nodata pixels -> nodata_mask
      4) Replace nodata with 0, scale reflectances => [0..1]
      5) Pass to OmniCloudMask => single-band classes in [0..3].
      6) For nodata_mask, set classification to 255
      7) Save classification mask => GeoTIFF with nodata=255
    Returns the output path (string) if successful, or None otherwise.
    """
    mission, tile, date_str, day_idx = key

    # 1) Build output file path; if it already exists => skip
    out_mask_name = f"{mission}_{tile}_{date_str}_{day_idx}_L2A_OmniCloudMask.tif"
    out_mask_path = os.path.join(data_dir, out_mask_name)

    if os.path.exists(out_mask_path):
        print(f"[INFO] OmniCloudMask already exists => {out_mask_path}. Skipping.")
        return None

    print(f"\n[SCENE] mission={mission}, tile={tile}, date={date_str}, day_idx={day_idx}")

    # 2) Check essential bands
    essential_bands = ["B04", "B03", "B08"]
    for eb in essential_bands:
        if eb not in band_dict:
            print("[WARNING] Missing essential band:", eb, "Skipping scene.")
            return None

    # 3) Read reference band (B04) => metadata
    ref_path = band_dict["B04"]
    try:
        with rasterio.open(ref_path) as ref_src:
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
            nodata_val = ref_src.nodata if (ref_src.nodata is not None) else -32768
            red_arr = ref_src.read(1)  # keep integer for now
    except Exception as e:
        print(f"[WARNING] Could not open {ref_path}, reason={e}")
        return None

    # 4) Read green + nir
    green_path = band_dict["B03"]
    nir_path = band_dict["B08"]
    try:
        with rasterio.open(green_path) as gsrc:
            green_arr = gsrc.read(1)
        with rasterio.open(nir_path) as nsrc:
            nir_arr = nsrc.read(1)
    except Exception as e:
        print(f"[WARNING] Could not open B03 or B08, reason={e}")
        return None

    # 5) Check dimensions
    if (green_arr.shape != red_arr.shape) or (nir_arr.shape != red_arr.shape):
        print("[ERROR] Mismatched band shapes; please align them before OmniCloudMask.")
        return None

    # 6) Build a combined nodata_mask
    nodata_mask = (
        (red_arr == nodata_val) |
        (green_arr == nodata_val) |
        (nir_arr == nodata_val)
    )

    # Convert these arrays to float32 for scaling
    red_arr   = red_arr.astype(np.float32)
    green_arr = green_arr.astype(np.float32)
    nir_arr   = nir_arr.astype(np.float32)

    # 7) Stack => shape=(3,H,W)
    stack_cpu = np.stack([red_arr, green_arr, nir_arr], axis=0)
    del red_arr, green_arr, nir_arr

    # 8) For non-nodata pixels, scale from 0..10000 => 0..1
    #    For nodata pixels, set them to 0.0 (so model sees 'blank')
    if USE_GPU:
        try:
            arr_gpu = cp.array(stack_cpu, dtype=cp.float32)
            arr_gpu = cp.where(arr_gpu == nodata_val, 0.0, arr_gpu)
            arr_gpu /= 10000.0
            stack_cpu = cp.asnumpy(arr_gpu)  # Move back to CPU for OmniCloudMask call
            del arr_gpu
        except cp.cuda.memory.OutOfMemoryError:
            print("   [WARNING] GPU OOM => fallback to CPU scaling.")
            stack_cpu = np.where(stack_cpu == nodata_val, 0.0, stack_cpu)
            stack_cpu /= 10000.0
        except Exception as e:
            print(f"   [WARNING] GPU error => fallback to CPU. Error: {e}")
            stack_cpu = np.where(stack_cpu == nodata_val, 0.0, stack_cpu)
            stack_cpu /= 10000.0
    else:
        # CPU-only
        stack_cpu = np.where(stack_cpu == nodata_val, 0.0, stack_cpu)
        stack_cpu /= 10000.0

    # 9) OmniCloudMask inference
    pred_mask = None
    if USE_GPU:
        # Try GPU
        try:
            pred_mask = run_omnimask_on_gpu(stack_cpu)
        except RuntimeError as e:
            err_str = str(e)
            if ("CUDA out of memory" in err_str) or ("CUDA error: out of memory" in err_str):
                print("   [WARNING] OmniCloudMask GPU OOM => fallback to CPU.")
                disable_gpu_globally()
                pred_mask = run_omnimask_on_cpu(stack_cpu)
            else:
                print(f"   [ERROR] OmniCloudMask GPU failed => {e}")
                return None
        except Exception as e:
            print(f"   [ERROR] OmniCloudMask GPU unexpected => {e}")
            return None
    else:
        # CPU-only
        print("   [OMNI] Running on CPU with batch_size=1...")
        try:
            pred_mask = run_omnimask_on_cpu(stack_cpu)
        except Exception as e:
            print(f"   [ERROR] OmniCloudMask CPU failed => {e}")
            return None

    if pred_mask is None:
        print("   [ERROR] No output from OmniCloudMask, skipping.")
        return None

    # 10) Ensure we have a (H,W) mask
    if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
        class_arr = pred_mask[0]
    else:
        class_arr = pred_mask

    class_arr = class_arr.astype(np.uint8)

    # 11) For previously nodata pixels, set classification to 255
    class_arr[nodata_mask] = 255

    # 12) Save single-band classification => GeoTIFF with nodata=255
    mask_img = Image(
        data=class_arr,
        transform=ref_transform,
        crs=ref_crs
    )
    try:
        mask_img.write_to_file(
            out_mask_path,
            dtype="uint8",
            nodata=255,    # Tag 255 as nodata
            compress="PACKBITS"
        )
    except Exception as e:
        print(f"   [ERROR] Failed to write OmniCloudMask output => {out_mask_path} : {e}")
        return None

    print(f"   [OMNI MASK] => {out_mask_path}")
    return out_mask_path

def main():
    # 1. Read config.json with UTF-8 encoding
    config_path = "config.json"
    if not os.path.exists(config_path):
        print("[ERROR] config.json not found. Exiting.")
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load config.json as UTF-8 => {e}")
        sys.exit(1)

    data_dir = config.get("cleaned_dir", None)
    if not data_dir or not os.path.isdir(data_dir):
        print("[ERROR] 'cleaned_dir' in config is invalid or does not exist.")
        sys.exit(1)

    # 2) Gather scenes => dictionary: { (mission,tile,date_str,day_idx): {band: path} }
    scene_map = {}
    all_files = os.listdir(data_dir)
    if not all_files:
        print(f"[ERROR] No files found in {data_dir}. Exiting.")
        sys.exit(0)

    for fname in all_files:
        # Only consider B03,B04,B08 .tif
        if not any(fname.endswith(bf) for bf in RELEVANT_BAND_FILES):
            continue

        mission, tile, date_str, day_idx = parse_s2_filename(fname)
        if mission is None:
            continue

        # Identify which band we are dealing with
        band_name = None
        for bfile in RELEVANT_BAND_FILES:
            if fname.endswith(bfile):
                band_name = bfile.replace(".tif", "")
                break
        if not band_name:
            continue

        key = (mission, tile, date_str, day_idx)
        full_path = os.path.join(data_dir, fname)
        if key not in scene_map:
            scene_map[key] = {}
        scene_map[key][band_name] = full_path

    if not scene_map:
        print("[ERROR] No relevant S2 bands (B03,B04,B08) found. Exiting.")
        sys.exit(0)

    # 3) Parallel processing
    #    If you have limited GPU memory, consider max_workers=1 to avoid OOM.
    max_workers = 1  # Use 1 for safer GPU usage; adjust if you have more memory
    print(f"[INFO] Found {len(scene_map)} scenes. Processing with {max_workers} worker(s)...")

    pred_paths = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for key, band_dict in scene_map.items():
            fut = executor.submit(process_scene, key, band_dict, data_dir)
            future_map[fut] = key

        for fut in as_completed(future_map):
            scene_key = future_map[fut]
            try:
                out_path = fut.result()
                if out_path:
                    pred_paths.append(out_path)
            except Exception as e:
                print(f"[ERROR] Scene {scene_key} => {e}")

    print("\n[DONE] Created OmniCloudMask outputs for all scenes.")
    if pred_paths:
        print("Resulting .tif files:")
        for p in pred_paths:
            print("   ", p)
    else:
        print("No new OmniCloudMask outputs were created.")

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    main()
