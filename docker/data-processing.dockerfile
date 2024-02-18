FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV AM_I_DOCKER True
ENV CUDA_HOME /usr/local/cuda-11.8/
ENV BUILD_WITH_CUDA 1
ENV TORCH_CUDA_ARCH_LIST="3.5 5.0 6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Install Python 3.9 and set it as the default Python version
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

RUN python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /home/appuser/
RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
WORKDIR /home/appuser/Grounded-Segment-Anything
RUN git clone https://github.com/mit-han-lab/efficientvit.git && cd efficientvit && pip install -e .
RUN python3 -m pip install segment-anything

# Llava v1.6
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
#RUN wget https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf
#RUN wget https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q3_K_M.gguf
#RUN wget https://huggingface.co/cjpais/llava-v1.6-34B-gguf/resolve/main/mmproj-model-f16.gguf
#RUN wget https://huggingface.co/cjpais/llava-v1.6-34B-gguf/resolve/main/llava-v1.6-34b.Q4_K_M.gguf

RUN python3 -m pip install --no-cache-dir wheel
RUN python3 -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO
RUN python3 -m pip install git+https://github.com/xinyu1205/recognize-anything.git

RUN wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l0.pt
RUN wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
RUN wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
RUN wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth

WORKDIR /home/appuser
RUN python3 -m pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
    pycocotools==2.0.6 matplotlib==3.5.3 \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai litellm

