FROM smellslikeml/vqasynth:base
WORKDIR /app

COPY docker/scene_fusion_stage/process_scene_fusion.py /app
COPY docker/scene_fusion_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN pip install pandas
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
