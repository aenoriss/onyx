# gsplat/splatfacto Layer for Onyx Pipeline
# Builds nerfstudio + gsplat from source for latest features
# Output: Standard PLY format (compatible with Unreal, web viewers, etc.)

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Build gsplat from source with correct CUDA architectures
# RTX 3060 = sm_86, RTX 4090 = sm_89
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"
ENV CUDA_ARCHITECTURES="86;89"
RUN pip install --no-cache-dir gsplat==1.4.0

# Install nerfstudio from source (has its own COLMAP parser, no pycolmap SceneManager needed)
RUN pip install --no-cache-dir nerfstudio

# Install additional dependencies
RUN pip install --no-cache-dir plyfile

# Create workspace
WORKDIR /workspace

# Copy training wrapper
COPY train.py /workspace/train.py

ENTRYPOINT ["python3", "/workspace/train.py"]
