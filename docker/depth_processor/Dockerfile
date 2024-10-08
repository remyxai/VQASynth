FROM smellslikeml/vqasynth:base
WORKDIR /app

RUN pip install timm==1.0.9
RUN pip install git+https://github.com/apple/ml-depth-pro.git

RUN mkdir -p checkpoints && \
    wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints

COPY docker/depth_processor/process_depth.py /app
COPY docker/depth_processor/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
