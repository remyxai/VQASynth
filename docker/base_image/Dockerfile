FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME /usr/local/cuda-11.8/

RUN apt-get update && apt-get install -y software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y build-essential git wget curl && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils python3-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    apt-get install -y zlib1g-dev libexpat1-dev

# Install CMake 3.26 or higher
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh && \
    chmod +x cmake-3.26.4-linux-x86_64.sh && \
    ./cmake-3.26.4-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.26.4-linux-x86_64.sh

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* vim=2:* -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

# Install yq for YAML processing
RUN wget https://github.com/mikefarah/yq/releases/download/v4.30.8/yq_linux_amd64 -O /usr/bin/yq \
    && chmod +x /usr/bin/yq

RUN pip install git+https://github.com/apple/ml-depth-pro.git

RUN pip install 'git+https://github.com/facebookresearch/sam2.git'
RUN pip install git+https://github.com/openai/CLIP.git

# Install PyTorch for CUDA 11.8 and other dependencies
RUN pip install --upgrade torch==2.4.0+cu118 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app/VQASynth
COPY . /app/VQASynth
RUN pip install .

# Install flash_attn and onnxruntime compatible with CUDA 11.8, order is important
RUN pip uninstall -y flash_attn
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.0

WORKDIR /app
