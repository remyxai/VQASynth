FROM vqasynth:base
WORKDIR /app

COPY docker/describe_anything_stage/process_describe_anything.py /app
COPY docker/describe_anything_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

# DAM ships as a pip-installable package. Weights download at runtime via
# transformers trust_remote_code (default: nvidia/DAM-3B-Self-Contained).
RUN pip install git+https://github.com/NVlabs/describe-anything
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
