FROM vqasynth:base
WORKDIR /app

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

COPY docker/location_refinement_stage/process_location_refinement.py /app
COPY docker/location_refinement_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
