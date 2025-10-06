FROM nvcr.io/nvidia/pytorch:24.08-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git ffmpeg libsm6 libxext6 libglib2.0-0 libgl1 \
        build-essential ninja-build cmake unzip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# Set CUDA_HOME for flash-attn compilation
ENV CUDA_HOME=/usr/local/cuda

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    pip install --no-build-isolation flash-attn==2.6.3.post1

COPY . /src

WORKDIR /src

ENV PYTHONPATH=/src \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/weights/hf \
    WAN_WEIGHTS_ROOT=/weights \
    MAX_JOBS=4

RUN mkdir -p /weights/hf
