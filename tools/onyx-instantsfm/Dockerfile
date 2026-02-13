# onyx-instantsfm: InstantSfM — COLMAP (GPU SIFT) + PyTorch 2.5.1
# Build with: docker build -f tools/onyx-instantsfm/Dockerfile -t onyx-instantsfm:latest tools/onyx-instantsfm
# Previous untested NGC version backed up as Dockerfile.ngc-untested

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System dependencies: COLMAP build deps + Ceres/suitesparse + OpenCV runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    # OpenCV / rendering
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # COLMAP build dependencies
    libboost-all-dev \
    libfreeimage-dev \
    libflann-dev \
    libsqlite3-dev \
    libglew-dev \
    libcgal-dev \
    # Ceres/suitesparse build deps
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libmetis-dev \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and create python alias
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip

# Install cuDSS (required by bae sparse solver)
ARG CUDSS_VERSION=0.7.1.4
RUN wget -q https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz -O /tmp/cudss.tar.xz \
    && tar -xJf /tmp/cudss.tar.xz -C /opt \
    && rm /tmp/cudss.tar.xz
ENV CUDSS_ROOT=/opt/libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive
ENV LD_LIBRARY_PATH=${CUDSS_ROOT}/lib:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDSS_ROOT}/lib:${LIBRARY_PATH}
ENV CPATH=${CUDSS_ROOT}/include:${CPATH}

# Build and install Ceres Solver (required by pyceres)
ARG CERES_VERSION=2.1.0
RUN wget -q http://ceres-solver.org/ceres-solver-${CERES_VERSION}.tar.gz \
    && tar zxf ceres-solver-${CERES_VERSION}.tar.gz \
    && mkdir ceres-build \
    && cd ceres-build \
    && cmake ../ceres-solver-${CERES_VERSION} -GNinja \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DMINIGLOG=OFF \
        -DSUITESPARSE=OFF \
        -DCXSPARSE=OFF \
    && ninja install \
    && cd / \
    && rm -rf ceres-build ceres-solver-${CERES_VERSION} ceres-solver-${CERES_VERSION}.tar.gz \
    && ldconfig

# Build COLMAP from source with GPU-accelerated SIFT
ARG COLMAP_VERSION=3.11.0
RUN git clone --branch ${COLMAP_VERSION} --depth 1 https://github.com/colmap/colmap.git /tmp/colmap \
    && mkdir /tmp/colmap/build && cd /tmp/colmap/build \
    && cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
        -DGUI_ENABLED=OFF \
        -DCUDA_ENABLED=ON \
        -DTESTS_ENABLED=OFF \
    && ninja \
    && ninja install \
    && rm -rf /tmp/colmap \
    && ldconfig

# Install PyTorch (CUDA 12.4 compatible wheels, runs on CUDA 12.6)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install pyceres (requires ceres-solver)
RUN pip install --no-cache-dir "pyceres @ git+https://github.com/cvg/pyceres.git@v2.6"

# Install scikit-sparse (requires suitesparse from apt)
RUN pip install --no-cache-dir scikit-sparse==0.4.15

# Install core Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0 \
    opencv-python==4.10.0.84 \
    pyyaml \
    matplotlib \
    tqdm==4.66.5 \
    kornia==0.8.0 \
    viser==0.2.11 \
    imageio-ffmpeg \
    "torchmetrics[image]" \
    tensorly \
    nerfview \
    tensorboard \
    scikit-learn \
    gradio \
    plotly \
    splines \
    jaxtyping \
    plyfile

# Install gsplat (CUDA kernels)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN pip install --no-cache-dir gsplat==1.5.3

# Install pypose (bae branch) and bae
RUN pip install --no-cache-dir "pypose @ git+https://github.com/pypose/pypose.git@bae"
RUN CPATH=${CUDSS_ROOT}/include:${CPATH} \
    LIBRARY_PATH=${CUDSS_ROOT}/lib:${LIBRARY_PATH} \
    pip install --no-cache-dir --no-build-isolation git+https://github.com/zitongzhan/bae.git

# NVIDIA container capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

# Copy InstantSfM source and install as editable package
COPY . /workspace/instantsfm
RUN cd /workspace/instantsfm && pip install --no-cache-dir -e . --no-deps

# Final ldconfig to ensure all shared libraries are indexed
RUN ldconfig

CMD ["/bin/bash"]
