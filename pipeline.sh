#!/usr/bin/env bash

# Sentinel-2 Monthly Mosaic Pipeline
# ==================================

set -e  # Exit on error

# Helper function to send Slack notifications (silent fail if not configured)
notify() {
    python notify_slack.py "$1" "$2" 2>/dev/null || true
}

# Track timing
PIPELINE_START=$(date +%s)

# Read dates from config.json
echo "[START] Sentinel-2 Pipeline"
echo "Reading configuration..."

# Extract dates from config.json using Python
read START_DATE END_DATE <<< $(python <<'EOF'
import json
with open("config.json", "r") as f:
    config = json.load(f)
print(config.get("start_data", "Unknown"), config.get("end_date", "Unknown"))
EOF
)

echo "Pipeline started for ${START_DATE} to ${END_DATE}"

# Send start notification
notify "Pipeline started for *${START_DATE}* to *${END_DATE}*" "start"

# 2. Download the data for the date range
echo ""
echo "[STEP 2] Downloading Sentinel-2 data..."
STEP_START=$(date +%s)
python downloadS2.py
STEP_END=$(date +%s)
STEP_DURATION=$((STEP_END - STEP_START))
notify "Download complete _(${STEP_DURATION}s)_" "download"

# 3. Clip the data to the bbox (parallel version)
echo ""
echo "[STEP 3] Clipping data to bounding box..."
STEP_START=$(date +%s)
python clipData_parallel.py
STEP_END=$(date +%s)
STEP_DURATION=$((STEP_END - STEP_START))
notify "Clipping complete _(${STEP_DURATION}s)_" "process"

# 4. Generate OmniCloudMasks
echo ""
echo "[STEP 4] Generating OmniCloudMasks..."
STEP_START=$(date +%s)
python omnimask.py
STEP_END=$(date +%s)
STEP_DURATION=$((STEP_END - STEP_START))
notify "Cloud masking complete _(${STEP_DURATION}s)_" "process"

# 5. Generate the mosaic
echo ""
echo "[STEP 5] Generating monthly mosaic..."
STEP_START=$(date +%s)
python generate_mosaic.py
STEP_END=$(date +%s)
STEP_DURATION=$((STEP_END - STEP_START))
notify "Mosaic generation complete _(${STEP_DURATION}s)_" "success"

# 6. Finalize: zip outputs, upload to Backblaze, notify via Slack
echo ""
echo "============================================================"
echo "  [STEP 6] Finalization: Backblaze upload + Slack notification"
echo "============================================================"
python finalize_and_upload.py

# Calculate total duration
PIPELINE_END=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END - PIPELINE_START))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "============================================================"
echo "  Pipeline Complete! (Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s)"
echo "============================================================"
