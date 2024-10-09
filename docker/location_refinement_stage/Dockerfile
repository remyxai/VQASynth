FROM smellslikeml/vqasynth:base
WORKDIR /app

COPY docker/location_refinement_stage/process_location_refinement.py /app
COPY docker/location_refinement_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
