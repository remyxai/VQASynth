FROM smellslikeml/vqasynth:base
WORKDIR /app

RUN pip install git+https://github.com/openai/CLIP.git

COPY process_filter.py /app
COPY entrypoint.sh /app

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
