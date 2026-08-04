FROM vqasynth:base
WORKDIR /app

COPY docker/detection_3d_stage/process_detection_3d.py /app
COPY docker/detection_3d_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN pip install pandas
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
