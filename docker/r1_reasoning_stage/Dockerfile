FROM vqasynth:base
WORKDIR /app

COPY docker/r1_reasoning_stage/process_reasoning.py /app
COPY docker/r1_reasoning_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN pip install pandas
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
