#!/bin/bash
# One-liner setup for RunPod
# Usage: curl -sSL https://raw.githubusercontent.com/aenoriss/onyx/main/deploy/runpod/setup.sh | bash

set -e

echo "=== Onyx Pipeline Setup ==="

# Download CLI
echo "[1/2] Downloading onyx CLI..."
curl -sSLO https://raw.githubusercontent.com/aenoriss/onyx/main/deploy/runpod/onyx
chmod +x onyx

# Pull image
echo "[2/2] Pulling Docker image (16.9GB)..."
./onyx pull

echo ""
echo "Setup complete! Run './onyx shell' to start."
