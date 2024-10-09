FROM smellslikeml/vqasynth:base
WORKDIR /app

RUN pip install git+https://github.com/openai/CLIP.git

COPY docker/filter_stage/process_filter.py /app
COPY docker/filter_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
