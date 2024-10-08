FROM smellslikeml/vqasynth:base
WORKDIR /app

# Install PyTorch for CUDA 11.8 and other dependencies
RUN pip install --upgrade torch==2.4.0+cu118 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers==4.44.2 huggingface-hub==0.24.7 accelerate==0.34.2 timm==1.0.9

# Install flash_attn compatible with CUDA 11.8, order is important
RUN pip uninstall -y flash_attn
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.0

COPY process_segment.py /app
COPY entrypoint.sh /app

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
