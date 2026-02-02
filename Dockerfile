# Use RunPod's optimized PyTorch image
# PyTorch 2.8.0 + CUDA 12.81 - FLUX.2-klein-4b-fp8 (~8GB VRAM, 1.6x faster than BF16)
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2204

WORKDIR /app

# Set environment variables
ENV MODEL_PATH=black-forest-labs/FLUX.2-klein-4b-fp8
ENV HF_HOME=/models/hf_cache
ENV PYTHONUNBUFFERED=1

# Install git and dependencies
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install diffusers from source for latest Flux2KleinPipeline support
# Pinned to specific commit for reproducibility (Jan 2026)
RUN pip install --no-cache-dir --no-deps git+https://github.com/huggingface/diffusers.git@4b64b5603ff9c8a690bd4ba62093636e6c6bcedc
RUN pip install --no-cache-dir regex requests filelock numpy Pillow

# Verify PyTorch version
RUN python -c "import torch; v=torch.__version__; print(f'PyTorch: {v}'); major_minor = tuple(map(int, v.split('+')[0].split('.')[:2])); assert major_minor >= (2,5), f'Need PyTorch 2.5+, got {v}'"

# Pre-download the FP8 model files (without loading - no GPU during build)
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('black-forest-labs/FLUX.2-klein-4b-fp8', \
    cache_dir='/models/hf_cache')"

# Copy handler
COPY handler.py .

# Run handler
CMD ["python", "-u", "handler.py"]
