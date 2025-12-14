#!/usr/bin/env python3
"""
notify_slack.py

Helper script to send Slack notifications from the pipeline.
Usage: python notify_slack.py "message" [status]

Arguments:
  message: The message to send
  status: Optional - "success", "error", "info" (default: "info")

Reads webhook URL from config.json -> slack.webhook_url
"""

import os
import sys
import json
from datetime import datetime

# Try to import requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import subprocess


def load_config(config_path="config.json"):
    """Load configuration from config.json"""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def send_slack_notification(webhook_url, message, status="info"):
    """Send notification to Slack via webhook"""
    if not webhook_url:
        print("[SLACK] Webhook URL not configured - skipping notification")
        return False
    
    # Color based on status (no emojis)
    status_config = {
        "success": {"color": "#36a64f", "prefix": "[SUCCESS]"},
        "error": {"color": "#ff0000", "prefix": "[ERROR]"},
        "warning": {"color": "#ffaa00", "prefix": "[WARNING]"},
        "info": {"color": "#2196f3", "prefix": "[INFO]"},
        "start": {"color": "#9c27b0", "prefix": "[START]"},
        "download": {"color": "#00bcd4", "prefix": "[DOWNLOAD]"},
        "process": {"color": "#ff9800", "prefix": "[PROCESS]"},
        "complete": {"color": "#4caf50", "prefix": "[COMPLETE]"},
    }
    
    cfg = status_config.get(status, status_config["info"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get hostname for identification
    hostname = os.environ.get("HOSTNAME", os.environ.get("VAST_CONTAINERLABEL", "unknown"))
    
    payload = {
        "attachments": [
            {
                "color": cfg["color"],
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"{cfg['prefix']} *Sentinel-2 Pipeline*\n{message}"
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Time: `{timestamp}` | Host: `{hostname[:20]}`"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    if HAS_REQUESTS:
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"[SLACK] Notification sent: {message[:50]}...")
                return True
            else:
                print(f"[SLACK] Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"[SLACK] Error: {e}")
            return False
    else:
        # Fallback to curl
        try:
            curl_cmd = [
                "curl", "-s", "-X", "POST",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(payload),
                webhook_url
            ]
            result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"[SLACK] Notification sent via curl: {message[:50]}...")
                return True
            return False
        except Exception as e:
            print(f"[SLACK] Curl error: {e}")
            return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python notify_slack.py \"message\" [status]")
        print("Status options: success, error, warning, info, start, download, process, complete")
        sys.exit(1)
    
    message = sys.argv[1]
    status = sys.argv[2] if len(sys.argv) > 2 else "info"
    
    config = load_config()
    slack_config = config.get("slack", {})
    
    if not slack_config.get("enabled", False):
        print("[SLACK] Notifications disabled in config")
        sys.exit(0)
    
    webhook_url = slack_config.get("webhook_url", "")
    
    if send_slack_notification(webhook_url, message, status):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
