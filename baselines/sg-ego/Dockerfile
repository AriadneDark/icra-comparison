#####################
# Development image #
#####################

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS builder

# Set non-interactive environment for apt-get to prevent hanging during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies and Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3-venv \
    git \
    ninja-build \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Map python command to python3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

# Upgrade pip and foundational packaging tools
# Create the virtual environment
RUN python -m venv /opt/venv

# Make sure we use the virtualenv!
# This is the equivalent of running 'source /opt/venv/bin/activate'
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

# Additional dependencies
RUN pip install --no-cache-dir causal-conv1d --no-build-isolation --no-deps
RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126 --no-deps


#################
# Runtime image #
#################

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# The virtualenv copied from the builder still needs a Python interpreter
# available in the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python

# Copy ONLY the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"