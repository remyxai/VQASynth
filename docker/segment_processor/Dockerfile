FROM smellslikeml/vqasynth:base
WORKDIR /app

RUN pip install timm flash_attn
RUN pip install 'git+https://github.com/facebookresearch/sam2.git'

COPY process_segment.py /app
COPY entrypoint.sh /app

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
