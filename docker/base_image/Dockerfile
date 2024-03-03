FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME /usr/local/cuda-11.8/

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9 && \
    python3 -m pip install --upgrade pip setuptools wheel

RUN apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* vim=2:* -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

WORKDIR /app/models
RUN wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l0.pt
RUN wget https://huggingface.co/cjpais/llava-v1.6-34B-gguf/resolve/main/mmproj-model-f16.gguf
RUN wget https://huggingface.co/cjpais/llava-v1.6-34B-gguf/resolve/main/llava-v1.6-34b.Q4_K_M.gguf

RUN python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app/VQASynth
COPY . /app/VQASynth
RUN pip install .
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.45 --force-reinstall --no-cache-dir

WORKDIR /app
