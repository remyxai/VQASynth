FROM smellslikeml/vqasynth:base
WORKDIR /app

COPY process_segment.py /app
COPY entrypoint.sh /app

RUN pip install pandas
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
