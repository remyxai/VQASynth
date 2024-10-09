FROM vqasynth:base
WORKDIR /app

RUN mkdir -p checkpoints && \
    wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints

COPY docker/depth_stage/process_depth.py /app
COPY docker/depth_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
