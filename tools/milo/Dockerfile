# onyx-milo: MILo Mesh-In-the-Loop Gaussian Splatting
# Build with: docker build -f tools/milo/Dockerfile -t onyx-milo:latest .
# (build context = repo root, so shared scripts are accessible)
# Output: High-quality triangulated mesh (.ply)
#
# Follows MILo's official install.py exactly

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Minimal system dependencies (EGL for nvdiffrast)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    # OpenCV / rendering
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # EGL for nvdiffrast
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Accept conda TOS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment (matches MILo's install.py)
RUN conda create -n milo python=3.9 -y

# Set CUDA environment for 11.8 (as per MILo instructions)
ENV CPATH=/usr/local/cuda/targets/x86_64-linux/include
ENV LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install PyTorch (exactly as install.py does)
RUN conda run -n milo conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 mkl=2023.1.0 -c pytorch -c nvidia

# Ninja required for JIT compilation of nvdiffrast CUDA extensions
RUN conda run -n milo conda install -y ninja

# Clone MILo (without --recursive, then fix SSH URLs)
WORKDIR /workspace
RUN git clone https://github.com/Anttwo/MILo.git && \
    cd MILo && \
    sed -i 's|git@github.com:|https://github.com/|g' .gitmodules && \
    git submodule sync && \
    git submodule update --init --recursive

WORKDIR /workspace/MILo

# Patch for Docker/headless: use CUDA context instead of OpenGL (MILo supports both)
# This changes the default from use_opengl=True to use_opengl=False
RUN sed -i 's/use_opengl=True/use_opengl=False/g' milo/scene/mesh.py

# Install requirements (as install.py does)
RUN conda run -n milo pip install -r requirements.txt

# Install submodules (--no-build-isolation to use installed torch)
RUN conda run -n milo pip install --no-build-isolation submodules/diff-gaussian-rasterization_ms
RUN conda run -n milo pip install --no-build-isolation submodules/diff-gaussian-rasterization
RUN conda run -n milo pip install --no-build-isolation submodules/diff-gaussian-rasterization_gof
RUN conda run -n milo pip install --no-build-isolation submodules/simple-knn
RUN conda run -n milo pip install --no-build-isolation submodules/fused-ssim

# Tetra-triangulation (exactly as install.py does)
WORKDIR /workspace/MILo/submodules/tetra_triangulation
RUN conda run -n milo conda install -y cmake
RUN conda run -n milo conda install -y conda-forge::gmp
RUN conda run -n milo conda install -y conda-forge::cgal
RUN conda run -n milo cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5
RUN conda run -n milo make
RUN conda run -n milo pip install --no-build-isolation -e .

# Nvdiffrast (editable install, --no-build-isolation for torch)
WORKDIR /workspace/MILo/submodules/nvdiffrast
RUN conda run -n milo pip install --no-build-isolation -e .

WORKDIR /workspace/MILo

# Copy training wrapper script
COPY tools/milo/train_wrapper.py /workspace/train_wrapper.py

# NVIDIA container capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_VISIBLE_DEVICES=all

WORKDIR /workspace

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "milo", "python", "/workspace/train_wrapper.py"]
