FROM smellslikeml/vqasynth:base
WORKDIR /app


COPY filter_dataset.py /app
COPY entrypoint.sh /app

RUN pip install pandas
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install datasets
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]

