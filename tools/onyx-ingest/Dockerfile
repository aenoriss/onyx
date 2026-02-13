# onyx-ingest: 360 Extractor + Standard Video Extractor
# Lightweight image — no COLMAP, no Ceres, no C++ compilation
# CUDA runtime only (for PyTorch/YOLOv8 GPU inference)

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Minimal system dependencies: OpenCV runtime + ffmpeg for telemetry extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    # OpenCV runtime
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # ffmpeg for video decoding + telemetry extraction
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create python alias and upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip

# Install PyTorch (CUDA 12.4 wheels, compatible with 12.6 runtime)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install 360Extractor dependencies
RUN pip install --no-cache-dir \
    opencv-python>=4.8.0 \
    numpy>=1.24.0 \
    ultralytics>=8.0.0 \
    piexif>=1.1.3 \
    PySide6>=6.5.0 \
    tqdm>=4.66.0

# NVIDIA container capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

# Copy 360Extractor source
# Build with: docker build -f tools/onyx-ingest/Dockerfile -t onyx-ingest:v2 .
COPY tools/360Extractor /workspace/360Extractor

CMD ["/bin/bash"]
