# SG-Ego runtime: upstream-pinned Torch 2.6 / CUDA 12.6 / Transformers 5.x.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    PYTHONPATH=/workspace/baselines/sg-ego

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ffmpeg git libgl1 libglib2.0-0 ninja-build \
        python3.12 python3.12-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

COPY baselines/sg-ego/requirements.txt /tmp/sg-ego-requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r /tmp/sg-ego-requirements.txt

# Keep a source snapshot in the image. Compose bind-mounts /workspace for local
# adaptations and generated outputs when running the benchmark.
COPY baselines/sg-ego /opt/sg-ego
COPY benchmark /opt/benchmark

WORKDIR /workspace
CMD ["python", "-c", "import torch; print('SG-Ego image ready; CUDA:', torch.cuda.is_available())"]

