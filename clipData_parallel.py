#!/usr/bin/env python3

import os
import sys
import json
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.ops import unary_union
from shapely import wkt
from concurrent.futures import ProcessPoolExecutor, as_completed


def clip_one_raster(
    input_file,
    input_dir,
    output_dir,
    wkt_geom,
    bbox_crs,
    bands_to_clip,
    remove_original=True
):
    """
    Worker function that clips a single raster (.tif) using a bounding geometry.
    Returns a tuple: (status_message, success_flag, input_path).
    """
    in_path = os.path.join(input_dir, input_file)

    # All files are now .tif, so keep the original filename
    out_file = input_file

    out_path = os.path.join(output_dir, out_file)

    # Skip if the clipped file already exists
    if os.path.exists(out_path):
        return f"[SKIP] {input_file} => Output already exists", False, in_path

    # Verify band name (if bands_to_clip is not empty)
    if bands_to_clip and not any(band in input_file for band in bands_to_clip):
        return f"[SKIP] {input_file} => Not in target bands", False, in_path

    # Convert WKT => shapely geometry
    bbox_geom = wkt.loads(wkt_geom)

    try:
        with rasterio.open(in_path) as src:
            # If needed, reproject bounding geometry to match the raster's CRS
            if bbox_crs and src.crs and (bbox_crs != src.crs):
                temp_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=bbox_crs)
                temp_gdf = temp_gdf.to_crs(src.crs)
                clip_geom = [unary_union(temp_gdf.geometry)]
            else:
                clip_geom = [bbox_geom]

            # Perform the mask (clip)
            out_image, out_transform = rasterio.mask.mask(
                src,
                clip_geom,
                crop=True,
                nodata=src.nodatavals[0] if src.nodatavals[0] is not None else None,
                all_touched=False
            )

            # Update metadata for the output file
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",  # Force output to GeoTIFF
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

        # Write clipped raster
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)

        # Verify the output file was created successfully
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            if remove_original:
                # Remove the original file to save space
                try:
                    os.remove(in_path)
                    return f"[OK] Clipped {input_file} -> {out_file} (original removed)", True, in_path
                except Exception as remove_error:
                    return f"[OK] Clipped {input_file} -> {out_file} (failed to remove original: {remove_error})", True, in_path
            else:
                return f"[OK] Clipped {input_file} -> {out_file}", True, in_path
        else:
            return f"[ERROR] {input_file} => Output file creation failed", False, in_path

    except Exception as e:
        return f"[ERROR] {input_file} => {e}", False, in_path


def main():
    # -------------------------------------------------------------------------
    # 1. Read config, get the bbox file
    # -------------------------------------------------------------------------
    config_file = "config.json"
    if not os.path.exists(config_file):
        print(f"Error: '{config_file}' not found.")
        sys.exit(1)

    with open(config_file, "r") as f:
        config = json.load(f)

    bbox_file = config.get("bbox")
    input_dir = config.get("download_dir")
    output_dir = config.get("cleaned_dir")
    bands_to_clip = config.get("bands_to_download", [])

    # Basic validation
    if not bbox_file or not os.path.exists(bbox_file):
        print(f"Error: bounding file '{bbox_file}' not found.")
        sys.exit(1)

    if not input_dir or not os.path.isdir(input_dir):
        print(f"Error: input directory '{input_dir}' not found.")
        sys.exit(1)

    if not output_dir:
        print("Error: no 'cleaned_dir' key found in config.json.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Load bounding geometry (GeoPandas) => unify => to WKT
    # -------------------------------------------------------------------------
    print(f"Reading bbox file: {bbox_file}")
    
    # Validate file format
    supported_formats = ['.gpkg', '.shp', '.geojson', '.json']
    file_ext = os.path.splitext(bbox_file)[1].lower()
    
    if file_ext not in supported_formats:
        print(f"Warning: File extension '{file_ext}' may not be supported.")
        print(f"Supported formats: {', '.join(supported_formats)}")
    else:
        print(f"Detected bbox format: {file_ext}")
    
    try:
        gdf = gpd.read_file(bbox_file)
    except Exception as e:
        print(f"Error reading bbox file '{bbox_file}': {e}")
        print("Ensure the file is a valid vector format (gpkg, shp, geojson)")
        sys.exit(1)
    
    if gdf.empty:
        print(f"Error: No features found in '{bbox_file}'.")
        sys.exit(1)
    
    print(f"Bbox CRS: {gdf.crs if gdf.crs else 'None'}")

    bbox_geom = unary_union(gdf.geometry)
    # We'll store the bounding geometry as WKT so it's picklable for multiprocessing
    bbox_wkt = bbox_geom.wkt
    bbox_crs = gdf.crs  # might be None

    # -------------------------------------------------------------------------
    # 3. Gather .tif files
    # -------------------------------------------------------------------------
    raster_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".tif")
    ]
    if not raster_files:
        print(f"No .tif files found in '{input_dir}'. Nothing to do.")
        return

    # -------------------------------------------------------------------------
    # 4. Clip in parallel
    # -------------------------------------------------------------------------
    print(
        f"[INFO] Found {len(raster_files)} raster files (.tif). Clipping in parallel...")

    max_workers = os.cpu_count() - 8
    results = []

    # Check if the user wants to disable file removal (for safety testing)
    remove_originals = config.get("remove_original_files_after_clipping", True)
    if not remove_originals:
        print("[INFO] File removal after clipping is DISABLED in config.")
    else:
        print("[INFO] Original files will be removed after successful clipping to save space.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                clip_one_raster,
                raster_file,
                input_dir,
                output_dir,
                bbox_wkt,
                bbox_crs,
                bands_to_clip,
                remove_originals
            ): raster_file
            for raster_file in raster_files
        }

        successful_clips = 0
        files_removed = 0
        
        for future in as_completed(future_to_file):
            raster_file = future_to_file[future]
            try:
                status_message, success_flag, input_path = future.result()
                results.append(status_message)
                
                if success_flag:
                    successful_clips += 1
                    if remove_originals and "original removed" in status_message:
                        files_removed += 1
                        
            except Exception as exc:
                msg = f"[ERROR] {raster_file} generated an exception: {exc}"
                results.append(msg)

    # Print summary
    print("\n[SUMMARY]")
    for r in results:
        print(r)

    print(f"\nClipping complete!")
    print(f"  - Successfully clipped: {successful_clips} files")
    print(f"  - Original files removed: {files_removed} files")
    print(f"  - Clipped GeoTIFFs are in: {output_dir}")
    
    if remove_originals and files_removed > 0:
        print(f"  - Saved disk space by removing {files_removed} original files")


if __name__ == "__main__":
    main()
