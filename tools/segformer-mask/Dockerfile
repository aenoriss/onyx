# onyx-segformer: SegFormer masks + optional Clean-GS Gaussian filtering
# Build with: docker build -f tools/segformer-mask/Dockerfile -t onyx-segformer:latest .
# (build context = repo root)

FROM ghcr.io/aenoriss/onyx-pipeline:v1

# Install only what we need for SegFormer + Clean-GS filtering
# transformers is already in base, but ensure correct version
# Pin transformers to version before CVE-2025-32434 check (base image has older PyTorch)
# plyfile + Pillow needed by gaussian_mask_filter.py (torch already in base)
RUN pip install --no-cache-dir \
    "transformers>=4.30,<4.47" \
    opencv-python \
    tqdm \
    plyfile

# Copy batch processing script and Clean-GS filter
COPY tools/segformer-mask/batch_mask.py /workspace/batch_mask.py
COPY tools/segformer-mask/gaussian_mask_filter.py /workspace/gaussian_mask_filter.py

# Model weights download on first run (~350MB)
# Pre-download skipped due to PyTorch/safetensors version requirements

WORKDIR /workspace

ENTRYPOINT ["python", "/workspace/batch_mask.py"]
