FROM vqasynth:base
WORKDIR /app

COPY docker/depth_stage/process_depth.py /app
COPY docker/depth_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
