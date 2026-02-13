#!/bin/bash
# MILo Setup Script for Thunder Compute
# Translates Dockerfile steps for non-Docker environment
# Matches: nvidia/cuda:11.8.0-devel-ubuntu22.04 base image

set -e

echo "=== Checking CUDA version ==="
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "WARNING: nvcc not found. CUDA may not be installed."
    echo "Thunder Compute should have CUDA pre-installed."
fi

echo "=== Installing system dependencies ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev

echo "=== Installing Miniconda ==="
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
fi

# Initialize conda for this script (matches how Dockerfile uses conda)
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda init bash

echo "=== Accepting conda TOS ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "=== Creating conda environment ==="
conda create -n milo python=3.9 -y || true

echo "=== Setting CUDA environment ==="
# These match the Dockerfile ENV exactly
export CPATH=/usr/local/cuda/targets/x86_64-linux/include
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Parallel compilation - use allocated vCPUs (default 16 for Thunder Compute)
# nproc may report host CPUs, not container allocation
NPROC=${THUNDER_VCPUS:-16}
export MAX_JOBS=$NPROC
export CMAKE_BUILD_PARALLEL_LEVEL=$NPROC
echo "Using $NPROC CPU cores for parallel compilation"

# Persist environment variables for future sessions
cat >> $HOME/.bashrc << 'ENVEOF'

# MILo environment (added by setup_thunder.sh)
export PATH="$HOME/miniconda3/bin:$PATH"
export CPATH=/usr/local/cuda/targets/x86_64-linux/include
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
export MAX_JOBS=${THUNDER_VCPUS:-16}
export CMAKE_BUILD_PARALLEL_LEVEL=${THUNDER_VCPUS:-16}
ENVEOF

echo "=== Installing PyTorch ==="
# Detect CUDA version and install matching PyTorch via pip (more reliable than conda)
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    echo "Detected CUDA $CUDA_VERSION - installing PyTorch with CUDA 12.4 via pip"
    conda run -n milo pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
else
    echo "Detected CUDA $CUDA_VERSION - installing PyTorch with CUDA 11.8 via pip"
    conda run -n milo pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
fi

echo "=== Installing Ninja ==="
conda run -n milo conda install -y ninja

echo "=== Cloning MILo ==="
cd $HOME
if [ ! -d "MILo" ]; then
    git clone https://github.com/Anttwo/MILo.git
    cd MILo
    sed -i 's|git@github.com:|https://github.com/|g' .gitmodules
    git submodule sync
    git submodule update --init --recursive
else
    cd MILo
    echo "MILo already cloned, skipping"
fi

echo "=== Patching for headless rendering ==="
# Exactly as Dockerfile: use_opengl=True -> use_opengl=False
sed -i 's/use_opengl=True/use_opengl=False/g' milo/scene/mesh.py

echo "=== Installing Python requirements ==="
conda run -n milo pip install -r requirements.txt

echo "=== Building CUDA extensions (this takes a while) ==="
# Exactly matching Dockerfile order and flags
conda run -n milo pip install --no-build-isolation submodules/diff-gaussian-rasterization_ms
conda run -n milo pip install --no-build-isolation submodules/diff-gaussian-rasterization
conda run -n milo pip install --no-build-isolation submodules/diff-gaussian-rasterization_gof

# CUDA 12.x fix: simple-knn needs cfloat header for FLT_MAX
if [ "$CUDA_MAJOR" -ge 12 ]; then
    echo "Applying CUDA 12.x compatibility patch for simple-knn..."
    if ! grep -q '#include <cfloat>' submodules/simple-knn/simple_knn.cu; then
        sed -i '1i #include <cfloat>' submodules/simple-knn/simple_knn.cu
    fi
fi
conda run -n milo pip install --no-build-isolation submodules/simple-knn
conda run -n milo pip install --no-build-isolation submodules/fused-ssim

echo "=== Building tetra-triangulation ==="
cd $HOME/MILo/submodules/tetra_triangulation
conda run -n milo conda install -y cmake
conda run -n milo conda install -y conda-forge::gmp
conda run -n milo conda install -y conda-forge::cgal
# Ensure CUDA headers are in include path for cmake
export CPATH=/usr/local/cuda/include:$CPATH
conda run -n milo cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
conda run -n milo make -j$NPROC
conda run -n milo pip install --no-build-isolation -e .

echo "=== Building nvdiffrast ==="
cd $HOME/MILo/submodules/nvdiffrast
conda run -n milo pip install --no-build-isolation -e .

echo "=== Copying train_wrapper.py ==="
cd $HOME/MILo
if [ -f "$HOME/train_wrapper.py" ]; then
    cp $HOME/train_wrapper.py $HOME/MILo/train_wrapper.py
    echo "train_wrapper.py copied to MILo directory"
fi

echo ""
echo "=========================================="
echo "=== Setup complete! ==="
echo "=========================================="
echo ""
echo "To use MILo:"
echo "  source ~/.bashrc"
echo "  conda activate milo"
echo "  cd ~/MILo"
echo ""
echo "Run training with:"
echo "  python train.py -s /path/to/scene"
echo ""
echo "Or use train_wrapper.py:"
echo "  python train_wrapper.py --scene /path/to/scene --metric outdoor"
