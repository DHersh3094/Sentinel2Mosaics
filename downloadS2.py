#!/usr/bin/env python3

import json
import sys
import os
import geopandas as gpd
from datetime import datetime
from pystac_client import Client
from urllib.parse import urlparse
import requests
import shutil
import time
import ssl
import urllib3
import concurrent.futures
import threading
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Suppress only the specific InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print a new line on complete
    if iteration == total:
        print()


def clear_line():
    """Clear the current line in the terminal."""
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns
    # Print spaces to clear the line, then return to the beginning
    print(' ' * terminal_width, end='\r')


def download_file(url, download_dir, item_id, band, current, total, max_retries=3, verify_ssl=True):
    """Download a file from a URL to the specified directory with custom naming."""
    # Create a custom filename format: "S2A_15SVU_20250318_1_L2A_B03.tif"
    try:
        # Custom filename with band from parameters
        filename = f"{item_id}_{band}.tif"
        filepath = os.path.join(download_dir, filename)
        
        # Check if file already exists and has content
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            clear_line()
            print_progress_bar(current, total, prefix=f'Overall Progress:',
                               suffix=f'({current}/{total}) Skipped (existing): {filename}', length=40)
            return True, filename

        # Clear previous download info and show progress
        clear_line()
        print_progress_bar(current, total, prefix=f'Overall Progress:',
                           suffix=f'({current}/{total}) Downloading: {filename}', length=40)

        # Create a session with retry logic
        session = create_session_with_retry()
        # Disable SSL verification if needed
        session.verify = verify_ssl

        # Try multiple times with increasing delays
        for attempt in range(max_retries):
            try:
                with session.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()

                    # Get file size for progress if available
                    file_size = int(r.headers.get('content-length', 0))
                    downloaded = 0

                    with open(filepath, 'wb') as f:
                        if file_size > 0:
                            # Show file download progress for larger files
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    # Update file download progress
                                    if downloaded % 51200 == 0:  # Update every ~50KB
                                        file_percent = int(
                                            100 * downloaded / file_size)
                                        clear_line()
                                        print_progress_bar(current, total,
                                                           prefix=f'Overall Progress:',
                                                           suffix=f'({current}/{total}) {filename}: {file_percent}%',
                                                           length=40)
                        else:
                            # No content length, just write the file
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                # If we got here, download was successful
                break

            except (requests.exceptions.SSLError, ssl.SSLError) as e:
                # Special handling for SSL errors - try without verification
                if attempt == max_retries - 1:
                    raise  # Re-raise the exception on the last attempt

                # Try again with SSL verification disabled
                verify_ssl = False
                clear_line()
                print_progress_bar(current, total, prefix=f'Overall Progress:',
                                   suffix=f'({current}/{total}) SSL Error, retrying without verification...',
                                   length=40)
                continue

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    raise  # Re-raise the exception on the last attempt

                # Wait with exponential backoff
                sleep_time = 2 ** attempt
                clear_line()
                print_progress_bar(current, total, prefix=f'Overall Progress:',
                                   suffix=f'({current}/{total}) Connection error, retrying in {sleep_time}s...',
                                   length=40)
                time.sleep(sleep_time)
                continue

        clear_line()
        print_progress_bar(current, total, prefix=f'Overall Progress:',
                           suffix=f'({current}/{total}) ✓ {filename}', length=40)
        return True, filename

    except Exception as e:
        clear_line()
        print_progress_bar(current, total, prefix=f'Overall Progress:',
                           suffix=f'({current}/{total}) ✗ Failed: {filename} - {str(e)[:30]}...', length=40)
        return False, None


# Global variables for tracking progress
progress_stats = {
    "downloaded": 0,
    "skipped": 0,
    "failed": 0,
    "current": 0,
    "total": 0
}
progress_lock = threading.Lock()

def update_progress(status="", increment_current=True):
    """Update the progress bar with current status"""
    with progress_lock:
        if increment_current:
            progress_stats["current"] += 1
            
        clear_line()
        print_progress_bar(
            progress_stats["current"], 
            progress_stats["total"], 
            prefix='Overall Progress:',
            suffix=f'({progress_stats["current"]}/{progress_stats["total"]}) '
                  f'↓{progress_stats["downloaded"]} '
                  f'⏭{progress_stats["skipped"]} '
                  f'✗{progress_stats["failed"]} {status}', 
            length=40
        )

def create_session_with_retry():
    """Create a requests session with retry logic"""
    session = requests.Session()

    # Configure retry strategy with more aggressive settings
    retry_strategy = Retry(
        total=5,  # Maximum number of retries
        backoff_factor=1,  # Will retry with 1s, 2s, 4s, 8s, 16s delays
        # Retry on these status codes
        status_forcelist=[429, 500, 502, 503, 504],
        # Retry only for these methods
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )

    # Increase connection pool for parallel downloads
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=50, pool_maxsize=50)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def parallel_download_file(url, download_dir, item_id, band, verify_ssl=True, max_retries=3):
    """Version of download_file optimized for parallel execution"""
    try:
        # Custom filename with band from parameters
        filename = f"{item_id}_{band}.tif"
        filepath = os.path.join(download_dir, filename)
        
        # Skip if file already exists and has content
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            with progress_lock:
                progress_stats["skipped"] += 1
            update_progress(f"Skipped: {filename}")
            return True, filename, "skipped"

        # Create a session with retry logic
        session = create_session_with_retry()
        # Disable SSL verification if needed
        session.verify = verify_ssl
        
        # Set longer timeout and headers to optimize download
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) Python/3 Sentinel-2-Downloader',
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=600, max=1000'
        }
        
        # Try multiple times with increasing delays
        for attempt in range(max_retries):
            try:
                with session.get(url, stream=True, timeout=180, headers=headers) as r:
                    r.raise_for_status()

                    # Write file in chunks with much larger buffer for maximum throughput
                    with open(filepath, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):  # 1MB chunks for better network utilization
                            if chunk:
                                f.write(chunk)
                
                # If we got here, download was successful
                with progress_lock:
                    progress_stats["downloaded"] += 1
                update_progress(f"Downloaded: {filename}")
                return True, filename, "downloaded"

            except (requests.exceptions.SSLError, ssl.SSLError) as e:
                # Try again with SSL verification disabled if SSL error
                if attempt == max_retries - 1:
                    raise
                verify_ssl = False
                continue

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    raise
                # Wait with exponential backoff
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
                continue

        return True, filename, "downloaded"

    except Exception as e:
        with progress_lock:
            progress_stats["failed"] += 1
        update_progress(f"Failed: {filename} - {str(e)[:30]}...")
        return False, None, f"error: {str(e)[:50]}"


def main():
    # 1. Read config.json
    with open("config.json", "r") as f:
        config = json.load(f)

    bbox_file = config.get("bbox")
    start_date = config.get("start_data")
    end_date = config.get("end_date")
    api_url = config.get(
        "endpoint", "https://earth-search.aws.element84.com/v1")
    collection = config.get("collection")
    download_dir = config.get("download_dir", "downloads")
    # Allow SSL verification to be disabled
    verify_ssl = config.get("verify_ssl", True)

    if not all([bbox_file, start_date, end_date, api_url, collection]):
        print("Error: 'config.json' must have 'bbox', 'start_data', 'end_date', 'endpoint', and 'collection'.")
        sys.exit(1)

    os.makedirs(download_dir, exist_ok=True)

    # 2. Read the geometry (supports .gpkg, .shp, .geojson)
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
    
    if gdf.crs is None:
        print("Warning: No CRS found. Assuming EPSG:4326.")
    else:
        print(f"Bbox CRS: {gdf.crs}")
        gdf = gdf.to_crs(epsg=4326)

    geom_union = gdf.unary_union
    time_range = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"

    # 4. Search STAC
    print(f"Connecting to STAC API: {api_url}")

    # Try connecting with multiple options if the default fails
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                # First attempt: Use standard connection
                client = Client.open(api_url)
                break
            elif attempt == 1:
                # Second attempt: Try with SSL verification disabled
                print("Retrying connection with SSL verification disabled...")
                # Create a custom STAC IO instance with SSL verification disabled
                from pystac.stac_io import DefaultStacIO
                from pystac_client.stac_api_io import StacApiIO

                custom_io = StacApiIO()
                custom_io.session.verify = False
                client = Client.open(api_url, stac_io=custom_io)
                break
            elif attempt == 2:
                # Third attempt: Try using a different STAC API endpoint
                alt_api_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
                print(f"Trying alternative STAC API: {alt_api_url}")
                client = Client.open(alt_api_url)
                break
        except Exception as e:
            print(f"Connection attempt {attempt+1} failed: {str(e)}")
            if attempt == max_attempts - 1:
                print(
                    "All connection attempts failed. Please check your network connection or API endpoint.")
                sys.exit(1)
            # Wait before retrying
            delay = 2 ** attempt
            print(f"Waiting {delay} seconds before retrying...")
            time.sleep(delay)

    print(
        f"Searching collection='{collection}' using geometry and datetime='{time_range}'")
    search = client.search(
        collections=[collection],
        intersects=geom_union.__geo_interface__,
        datetime=time_range
    )

    items = list(search.get_items())
    print(f"Found {len(items)} items from STAC search.")

    # Get the list of bands to download from config
    bands_to_download = config.get("bands_to_download", [])
    print(f"Will download only these bands: {bands_to_download}")

    # Map band codes to STAC asset names for TIF files
    band_to_asset_map = {
        "B02": "blue",    # Blue band
        "B03": "green",   # Green band
        "B04": "red",     # Red band
        "B08": "nir",     # Near Infrared (NIR) band
        "SCL": "scl",     # Scene Classification Layer
        "TCI": "visual"   # True color image
    }

    print(
        f"Using this mapping from band codes to STAC assets: {band_to_asset_map}")

    # Debug: Print HREF types for the first item to see what we're dealing with
    if items:
        print("\nAvailable assets in the first item:")
        for asset_key, asset in items[0].assets.items():
            href = asset.get_absolute_href()
            print(
                f"  - {asset_key}: {href[:60]}{'...' if len(href) > 60 else ''}")

        print("\nLooking for these asset names:")
        for band in bands_to_download:
            if band in band_to_asset_map:
                print(f"  - Band '{band}' → Asset '{band_to_asset_map[band]}'")
            else:
                print(f"  - Band '{band}' → No mapping found!")

    # Calculate the total number of potential downloads
    total_potential_downloads = len(items) * len(bands_to_download)
    print(f"\nTotal potential downloads: {total_potential_downloads}")
    
    # Get the number of workers from config or use CPU-based default
    max_workers = os.cpu_count() - 16 
    print(f"Using {max_workers} parallel download workers")
    
    # Build a list of download tasks
    download_tasks = []
    
    for item in items:
        item_id = item.id
        
        for band in bands_to_download:
            # Map band code to STAC asset name
            asset_key = band_to_asset_map.get(band)
            
            if not asset_key:
                continue
                
            if asset_key in item.assets:
                asset = item.assets[asset_key]
                href = asset.get_absolute_href()
                
                if (href and 
                    href.endswith((".tif", ".tiff")) and 
                    href.startswith(("http://", "https://"))):
                    
                    download_tasks.append((href, download_dir, item_id, band, verify_ssl))
    
    # Initialize progress tracking
    progress_stats["total"] = len(download_tasks)
    progress_stats["current"] = 0
    update_progress("Starting downloads...", increment_current=False)
    
    url_list = []
    start_time = time.time()
    
    # Execute downloads in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_download = {
            executor.submit(
                parallel_download_file, url, dir_path, item_id, band, verify_ssl
            ): (url, item_id, band) 
            for url, dir_path, item_id, band, verify_ssl in download_tasks
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_download):
            url, item_id, band = future_to_download[future]
            try:
                success, filename, status = future.result()
                if success and status != "skipped":
                    url_list.append(url)
            except Exception as exc:
                print(f'\nERROR: {item_id}_{band} generated an exception: {exc}')
    
    # Finish progress bar
    update_progress("Complete!", increment_current=False)
    
    # Calculate elapsed time and download rate
    elapsed_time = time.time() - start_time
    download_rate = progress_stats["downloaded"] / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\nDownload summary:")
    print(f"  - Successfully downloaded: {progress_stats['downloaded']} files")
    print(f"  - Skipped (already exists): {progress_stats['skipped']} files")
    print(f"  - Failed: {progress_stats['failed']} files")
    print(f"  - Total URLs collected: {len(url_list)}")
    print(f"  - Total time: {elapsed_time:.1f} seconds")
    print(f"  - Download rate: {download_rate:.2f} files/second")

    try:
        year_str = str(datetime.strptime(start_date, "%Y-%m-%d").year)
    except ValueError:
        year_str = "YYYY"

    copernicus_notice = (
        "Contains modified Copernicus Sentinel data [{}] for Sentinel data. "
        "Contains modified Copernicus Service information [{}] for Copernicus Service Information."
    ).format(year_str, year_str)

    # 7. Write metadata.json
    output_data = {
        "timeframe": {
            "start_date": start_date,
            "end_date": end_date
        },
        "urlList": url_list,
        "copernicusNotice": copernicus_notice
    }

    with open("metadata.json", "w") as out_file:
        json.dump(output_data, out_file, indent=2)

    print("metadata.json written and files downloaded to:", download_dir)


if __name__ == "__main__":
    main()
