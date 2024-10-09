FROM vqasynth:base
WORKDIR /app

COPY docker/embeddings_stage/process_embeddings.py /app
COPY docker/embeddings_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
