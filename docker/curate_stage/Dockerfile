FROM vqasynth:base
WORKDIR /app

COPY docker/curate_stage/process_curate.py /app
COPY docker/curate_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
