#!/usr/bin/env python3
"""
finalize_and_upload.py

This script:
1. Zips all OmniCloudMask files and output files (mosaic)
2. Uploads the zip files and main outputs to Backblaze B2
3. Sends a Slack notification via webhook

Configuration is read from config.json:
- backblaze.application_key_id: Backblaze B2 application key ID
- backblaze.application_key: Backblaze B2 application key
- backblaze.bucket_name: Target bucket name
- backblaze.bucket_prefix: Prefix/folder in bucket (optional)
- slack.webhook_url: Slack webhook URL for notifications
"""

import os
import sys
import json
import glob
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path

# Try to import requests for Slack webhook
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("[WARNING] 'requests' library not installed. Slack notifications will use curl fallback.")

# Try to import b2sdk for Backblaze
try:
    from b2sdk.v2 import B2Api, InMemoryAccountInfo
    HAS_B2SDK = True
except ImportError:
    HAS_B2SDK = False
    print("[WARNING] 'b2sdk' library not installed. Will use b2 CLI fallback.")


def load_config(config_path="config.json"):
    """Load configuration from config.json"""
    if not os.path.exists(config_path):
        print(f"[ERROR] {config_path} not found.")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(metadata_path="metadata.json"):
    """Load metadata from metadata.json if it exists"""
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_date_string(config):
    """Extract date string from config for file naming"""
    start_date_str = config.get("start_data", config.get("start_date", ""))
    end_date_str = config.get("end_date", None)
    
    if start_date_str:
        try:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
            
            # Set end date to end of month if not specified
            if end_date_str:
                end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
            else:
                # Default to end of month if no end date specified
                from datetime import timedelta
                if start_dt.month == 12:
                    end_dt = datetime(start_dt.year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_dt = datetime(start_dt.year, start_dt.month + 1, 1) - timedelta(days=1)
            
            # Create date range string matching generate_mosaic.py format
            return f"{start_dt.strftime('%Y-%m-%d')}_to_{end_dt.strftime('%Y-%m-%d')}"
        except ValueError:
            pass
    
    # Fallback
    return datetime.now().strftime("%Y-%m-%d_to_%Y-%m-%d")


def find_omnimask_files(cleaned_dir):
    """Find all OmniCloudMask.tif files in the cleaned directory"""
    if not os.path.isdir(cleaned_dir):
        print(f"[WARNING] Cleaned directory not found: {cleaned_dir}")
        return []
    
    pattern = os.path.join(cleaned_dir, "*OmniCloudMask.tif")
    files = glob.glob(pattern)
    print(f"[INFO] Found {len(files)} OmniCloudMask files")
    return files


def find_output_files(output_dir, date_str):
    """Find mosaic files in output directory"""
    files = []
    if not os.path.isdir(output_dir):
        print(f"[WARNING] Output directory not found: {output_dir}")
        return files
    
    # Look for mosaic files
    patterns = [
        f"*_{date_str}.tif",      # Main mosaic
        f"*_{date_str}.vrt",      # VRT file
        f"*_{date_str}_*.tif",    # Any related files
    ]
    
    for pattern in patterns:
        found = glob.glob(os.path.join(output_dir, pattern))
        for f in found:
            if f not in files:
                files.append(f)
    
    # Include metadata.json in the output files
    metadata_path = "metadata.json"
    if os.path.exists(metadata_path) and metadata_path not in files:
        files.append(metadata_path)
        print(f"[INFO] Including metadata.json in output files")
    
    print(f"[INFO] Found {len(files)} output files")
    return files


def create_zip_archive(files, zip_path, base_dir=None):
    """Create a zip archive containing the specified files"""
    if not files:
        print(f"[WARNING] No files to zip for {zip_path}")
        return None
    
    print(f"[INFO] Creating zip archive: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filepath in files:
            if os.path.exists(filepath):
                # Use relative path in archive
                arcname = os.path.basename(filepath)
                print(f"  Adding: {arcname}")
                zipf.write(filepath, arcname)
            else:
                print(f"  [WARNING] File not found: {filepath}")
    
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"[INFO] Created {zip_path} ({size_mb:.2f} MB)")
        return zip_path
    return None


def upload_to_backblaze_sdk(b2_config, files_to_upload):
    """Upload files to Backblaze B2 using b2sdk"""
    key_id = b2_config.get("application_key_id", "")
    app_key = b2_config.get("application_key", "")
    bucket_name = b2_config.get("bucket_name", "")
    prefix = b2_config.get("bucket_prefix", "")
    
    if not all([key_id, app_key, bucket_name]):
        print("[ERROR] Backblaze credentials not configured in config.json")
        return False
    
    try:
        # Initialize B2 API
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", key_id, app_key)
        
        # Get bucket
        bucket = b2_api.get_bucket_by_name(bucket_name)
        
        uploaded_files = []
        for filepath in files_to_upload:
            if not os.path.exists(filepath):
                print(f"  [WARNING] File not found: {filepath}")
                continue
            
            filename = os.path.basename(filepath)
            remote_path = f"{prefix}/{filename}" if prefix else filename
            
            print(f"  Uploading: {filename} -> b2://{bucket_name}/{remote_path}")
            
            bucket.upload_local_file(
                local_file=filepath,
                file_name=remote_path,
            )
            uploaded_files.append(remote_path)
            print(f"  [OK] Uploaded: {filename}")
        
        return uploaded_files
        
    except Exception as e:
        print(f"[ERROR] Backblaze SDK upload failed: {e}")
        return False


def upload_to_backblaze_cli(b2_config, files_to_upload):
    """Upload files to Backblaze B2 using b2 CLI (fallback)"""
    key_id = b2_config.get("application_key_id", "")
    app_key = b2_config.get("application_key", "")
    bucket_name = b2_config.get("bucket_name", "")
    prefix = b2_config.get("bucket_prefix", "")
    
    if not all([key_id, app_key, bucket_name]):
        print("[ERROR] Backblaze credentials not configured in config.json")
        return False
    
    try:
        # Authorize with B2 CLI
        print("[INFO] Authorizing with Backblaze B2...")
        auth_cmd = ["b2", "authorize-account", key_id, app_key]
        result = subprocess.run(auth_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] B2 authorization failed: {result.stderr}")
            return False
        
        uploaded_files = []
        for filepath in files_to_upload:
            if not os.path.exists(filepath):
                print(f"  [WARNING] File not found: {filepath}")
                continue
            
            filename = os.path.basename(filepath)
            remote_path = f"{prefix}/{filename}" if prefix else filename
            
            print(f"  Uploading: {filename} -> b2://{bucket_name}/{remote_path}")
            
            upload_cmd = ["b2", "upload-file", bucket_name, filepath, remote_path]
            result = subprocess.run(upload_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                uploaded_files.append(remote_path)
                print(f"  [OK] Uploaded: {filename}")
            else:
                print(f"  [ERROR] Upload failed for {filename}: {result.stderr}")
        
        return uploaded_files
        
    except Exception as e:
        print(f"[ERROR] Backblaze CLI upload failed: {e}")
        return False


def upload_to_backblaze(b2_config, files_to_upload):
    """Upload files to Backblaze B2"""
    if HAS_B2SDK:
        return upload_to_backblaze_sdk(b2_config, files_to_upload)
    else:
        return upload_to_backblaze_cli(b2_config, files_to_upload)


def send_slack_notification(webhook_url, message, success=True):
    """Send notification to Slack via webhook"""
    if not webhook_url:
        print("[WARNING] Slack webhook URL not configured")
        return False
    
    # Build the Slack message payload
    color = "#36a64f" if success else "#ff0000"
    status_prefix = "[SUCCESS]" if success else "[ERROR]"
    
    payload = {
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"{status_prefix} *Sentinel-2 Mosaic Pipeline*\n{message}"
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    if HAS_REQUESTS:
        try:
            response = requests.post(webhook_url, json=payload, timeout=30)
            if response.status_code == 200:
                print("[INFO] Slack notification sent successfully")
                return True
            else:
                print(f"[WARNING] Slack notification failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"[WARNING] Slack notification error: {e}")
            return False
    else:
        # Fallback to curl
        try:
            import json as json_module
            curl_cmd = [
                "curl", "-X", "POST",
                "-H", "Content-Type: application/json",
                "-d", json_module.dumps(payload),
                webhook_url
            ]
            result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("[INFO] Slack notification sent via curl")
                return True
            else:
                print(f"[WARNING] Slack curl failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"[WARNING] Slack curl error: {e}")
            return False


def main():
    print("\n" + "="*60)
    print("  Sentinel-2 Mosaic Finalization & Upload Script")
    print("="*60 + "\n")
    
    # Load configuration
    config = load_config()
    metadata = load_metadata()
    
    # Get date string for file naming
    date_str = get_date_string(config)
    print(f"[INFO] Processing date: {date_str}")
    
    # Get output filename base from config
    output_base = config.get("output_filename_base", "Mosaic")
    print(f"[INFO] Output filename base: {output_base}")
    
    # Get directories from config
    cleaned_dir = config.get("cleaned_dir", "../MosaicData/cleaned_downloaded_data")
    output_dir = config.get("output_dir", "../MosaicData/outputs")
    
    # Get configuration sections
    b2_config = config.get("backblaze", {})
    slack_config = config.get("slack", {})
    
    # Track files to upload
    files_to_upload = []
    
    # Step 1: Find and zip OmniCloudMask files
    print("\n[STEP 1] Processing OmniCloudMask files...")
    omnimask_files = find_omnimask_files(cleaned_dir)
    
    if omnimask_files:
        omnimask_zip = os.path.join(output_dir, f"{output_base}_OmniCloudMasks_{date_str}.zip")
        os.makedirs(output_dir, exist_ok=True)
        zip_result = create_zip_archive(omnimask_files, omnimask_zip)
        if zip_result:
            files_to_upload.append(zip_result)
    
    # Step 2: Find output files (mosaic) and zip them
    print("\n[STEP 2] Processing output files...")
    output_files = find_output_files(output_dir, date_str)
    
    # Only create and upload zip of all outputs (no individual TIF files)
    if output_files:
        outputs_zip = os.path.join(output_dir, f"{output_base}_AllOutputs_{date_str}.zip")
        zip_result = create_zip_archive(output_files, outputs_zip)
        if zip_result:
            files_to_upload.append(zip_result)
            print(f"[INFO] Created output zip containing {len(output_files)} files")
    
    print(f"\n[INFO] Total files to upload: {len(files_to_upload)}")
    for f in files_to_upload:
        print(f"  - {os.path.basename(f)}")
    
    # Step 3: Upload to Backblaze
    upload_success = False
    uploaded_files = []
    
    if b2_config.get("enabled", False):
        print("\n[STEP 3] Uploading to Backblaze B2...")
        result = upload_to_backblaze(b2_config, files_to_upload)
        if result:
            upload_success = True
            uploaded_files = result if isinstance(result, list) else []
            print(f"[INFO] Successfully uploaded {len(uploaded_files)} files to Backblaze")
        else:
            print("[ERROR] Backblaze upload failed!")
    else:
        print("\n[STEP 3] Backblaze upload disabled in config")
    
    # Step 4: Send Slack notification
    if slack_config.get("enabled", False) and slack_config.get("webhook_url"):
        print("\n[STEP 4] Sending Slack notification...")
        
        # Build notification message
        bucket_name = b2_config.get("bucket_name", "unknown")
        if upload_success:
            message = f"*Upload Complete!*\n\n"
            message += f"*Files uploaded:* {len(uploaded_files)}\n"
            message += f"*Bucket:* `{bucket_name}`\n"
            message += f"*Period:* {date_str}\n\n"
            message += "*Uploaded files:*\n"
            for f in uploaded_files[:5]:  # Show first 5
                message += f"- `{os.path.basename(f) if isinstance(f, str) else f}`\n"
            if len(uploaded_files) > 5:
                message += f"_...and {len(uploaded_files) - 5} more files_"
        else:
            message = f"*Upload Failed or Skipped*\n\n"
            message += f"*Period:* {date_str}\n"
            message += "Please check the logs for details."
        
        send_slack_notification(slack_config["webhook_url"], message, upload_success)
    else:
        print("\n[STEP 4] Slack notifications disabled in config")
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  Files processed: {len(files_to_upload)}")
    print(f"  Upload success: {upload_success}")
    print(f"  Slack notified: {slack_config.get('enabled', False)}")
    print("="*60 + "\n")
    
    return 0 if upload_success or not b2_config.get("enabled", False) else 1


if __name__ == "__main__":
    sys.exit(main())
