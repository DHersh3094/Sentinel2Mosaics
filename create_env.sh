#!/usr/bin/env bash
set -e

# 1) If conda is not found, install Miniconda
if ! command -v conda &>/dev/null; then
  echo "[INFO] Conda not detected. Installing Miniconda..."

  # Download latest Miniconda installer (Linux x86_64)
  # Adjust if you're on macOS or a different architecture
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

  # Run the installer silently (-b) to ~/miniconda
  bash /tmp/miniconda.sh -b -p $HOME/miniconda

  # Initialize conda (writes to ~/.bashrc by default)
  $HOME/miniconda/bin/conda init bash
  
  # Add conda to PATH for current session
  export PATH="$HOME/miniconda/bin:$PATH"

  echo "[INFO] Miniconda installed to ~/miniconda."
  echo "[INFO] Added conda to PATH for current session."
fi

# 2) Ensure conda is on PATH for this script
#    Typically added by 'conda init', but we can also source it explicitly:
if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
else
  # Try additional common locations
  for CONDA_PATH in "/usr/local/conda" "/opt/miniconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
      source "$CONDA_PATH/etc/profile.d/conda.sh"
      break
    fi
  done
  
  # Last resort: add to PATH directly if we can find the binary
  for CONDA_BIN in "$HOME/miniconda/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda"; do
    if [ -f "$CONDA_BIN" ]; then
      export PATH="$(dirname "$CONDA_BIN"):$PATH"
      break
    fi
  done
  
  # Check if conda is now available
  if ! command -v conda &>/dev/null; then
    echo "[WARNING] Could not find conda. Make sure conda is in your PATH."
    echo "          You may need to open a new shell after installation."
  else
    echo "[INFO] Found conda at $(which conda)"
  fi
fi

# 3) Now 'conda' should be available. Create 'mosaics' environment:
conda create --name mosaics -c conda-forge -c defaults \
    python=3.10 \
    gdal \
    pip \
    cudatoolkit \
    -y

# 4) Activate the newly created environment
echo "[INFO] Activating mosaics environment..."
eval "$(conda shell.bash hook)"
conda activate mosaics

# Verify environment activation
if [[ "$(conda info --envs | grep '*' | awk '{print $1}')" != "mosaics" ]]; then
  echo "[ERROR] Failed to activate mosaics environment. Pip installations may fail."
  exit 1
else
  echo "[INFO] Successfully activated mosaics environment."
fi

# 5) Install additional Python libraries via pip
#    - Includes 'awscli', 'pystac-client', 'geopandas', etc.
echo "[INFO] Installing Python packages via pip..."
pip install \
    rasterio \
    shapely \
    folium \
    cupy-cuda12x \
    dask \
    omnicloudmask \
    geopandas \
    attrs \
    awscli \
    pystac-client \
    rasterstats \
    ukis-pysat[raster] \
    b2sdk \
    requests --upgrade

# 6) Install Backblaze B2 CLI tool
echo "[INFO] Installing Backblaze B2 CLI..."
pip install b2 --upgrade

# Verify B2 CLI installation
if command -v b2 &>/dev/null; then
    echo "[INFO] Backblaze B2 CLI installed successfully: $(b2 version 2>/dev/null || echo 'version check unavailable')"
else
    echo "[WARNING] B2 CLI may not be in PATH. Try running: pip show b2"
fi

# Try to set locale, but don't fail if it's not available
if locale -a | grep -q "en_US.UTF-8"; then
  export LANG=en_US.UTF-8
  export LC_ALL=en_US.UTF-8
else
  echo "[WARNING] en_US.UTF-8 locale not available. Using system default locale."
fi

# 6) Confirm installation
echo ""
echo "----------------------------------------------------------"
echo "Environment 'mosaics' has been created with additional packages."
echo "Use 'conda activate mosaics' to enter the environment."
echo ""
echo "IMPORTANT: If 'conda activate mosaics' fails with 'command not found',"
echo "           run the following commands to add conda to your PATH:"
echo ""
echo "  source ~/.bashrc"
echo "  # or"
echo "  export PATH=\"\$HOME/miniconda/bin:\$PATH\""
echo "----------------------------------------------------------"
