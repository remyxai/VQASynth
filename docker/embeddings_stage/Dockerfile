FROM smellslikeml/vqasynth:base
WORKDIR /app

RUN pip install git+https://github.com/openai/CLIP.git

COPY docker/embeddings_stage/process_embeddings.py /app
COPY docker/embeddings_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
