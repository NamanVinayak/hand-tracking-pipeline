# HaMeR RunPod Serverless Docker Image
# Based on official HaMeR docker/hamer-dev.Dockerfile

ARG BASE=nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM ${BASE}

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install OS dependencies (from official HaMeR Dockerfile)
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    gcc g++ \
    make \
    python3 python3-dev python3-pip python3-venv python3-wheel \
    espeak-ng libsndfile1-dev \
    git \
    wget \
    ffmpeg \
    libsm6 libxext6 \
    libglfw3-dev libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
WORKDIR /app
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and setuptools
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install numpy separately (HaMeR requirement)
RUN pip install numpy

# Install gdown for model downloads
RUN pip install gdown

# Install chumpy and other dependencies that have build issues
# chumpy uses 'import pip' in setup.py which fails with build isolation
RUN pip install --no-build-isolation chumpy && \
    pip install scipy opencv-python scikit-image

# Clone HaMeR repository
RUN git clone --recursive https://github.com/geopavlakos/hamer.git

# Install ViTPose with --no-build-isolation to use existing packages
WORKDIR /app/hamer
RUN pip install --no-build-isolation -v -e third-party/ViTPose

# Install HaMeR with all dependencies
# detectron2 needs torch during build, so use --no-build-isolation
RUN pip install --no-build-isolation -e .[all]

# Download HaMeR checkpoints (6GB - baked into image)
RUN bash fetch_demo_data.sh

# Install RunPod and boto3 for serverless + S3
RUN pip install runpod boto3

# Copy our custom handler
COPY handler.py /app/handler.py

# Fix file permissions for RunPod (avoid user namespace mapping errors)
RUN chown -R root:root /app /opt/venv

# Set working directory
WORKDIR /app

# RunPod serverless entry point
CMD ["python", "-u", "handler.py"]
