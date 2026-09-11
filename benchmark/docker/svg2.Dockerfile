# SVG2 annotation runtime. Training-only DeepSpeed/FlashAttention are omitted.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    SAM2_BUILD_CUDA=0 \
    PYTHONPATH=/workspace/baselines/svg2

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ffmpeg git libgl1 libglib2.0-0 ninja-build \
        python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir \
        'transformers==4.54.1' 'accelerate>=1.4' 'huggingface_hub>=0.34' \
        sentencepiece protobuf einops timm hydra-core iopath 'openai>=1.40' \
        'pycocotools>=2.0.7' 'opencv-python-headless>=4.9' 'decord>=0.6' \
        numpy 'pillow>=10' 'pydantic>=2.5' 'pyyaml>=6' 'tqdm>=4.66' \
    && python -m pip install --no-cache-dir \
        'git+https://github.com/facebookresearch/sam2.git' \
    && python -m pip install --no-cache-dir --no-deps \
        'git+https://github.com/NVlabs/describe-anything.git'

COPY baselines/svg2 /opt/svg2
COPY benchmark /opt/benchmark

WORKDIR /workspace
CMD ["python", "-c", "import torch; print('SVG2 image ready; CUDA:', torch.cuda.is_available())"]

