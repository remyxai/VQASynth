FROM vqasynth:base
WORKDIR /app

# opencv-python ships in the base image (via requirements.txt), so no extra
# install is needed here. See docker/base_image/Dockerfile (`pip install .`).
COPY docker/correspondence_stage/process_correspondence.py /app
COPY docker/correspondence_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
