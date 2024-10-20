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

RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh && \
    chmod +x cmake-3.26.4-linux-x86_64.sh && \
    ./cmake-3.26.4-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.26.4-linux-x86_64.sh

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN python3 -m pip install --upgrade pip && python3 -m pip install setuptools==65.0.1 wheel spacy==3.7.5
RUN python3 -m spacy download en_core_web_sm

RUN python3 -m pip install numpy==1.21.0
RUN python3 -m pip install scikit-learn==1.0.2 --prefer-binary

RUN apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* vim=2:* -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/mikefarah/yq/releases/download/v4.30.8/yq_linux_amd64 -O /usr/bin/yq \
    && chmod +x /usr/bin/yq

RUN pip install git+https://github.com/apple/ml-depth-pro.git
RUN pip install 'git+https://github.com/facebookresearch/sam2.git'
RUN pip install git+https://github.com/openai/CLIP.git

RUN pip install --upgrade torch==2.4.0+cu118 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app/VQASynth
COPY . /app/VQASynth
RUN pip install .

RUN pip uninstall -y flash_attn
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.0

WORKDIR /app

