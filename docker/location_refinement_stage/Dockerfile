FROM vqasynth:base
WORKDIR /app

WORKDIR /app/checkpoints
RUN wget https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2/sam2_hiera_l.yaml
RUN wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

WORKDIR /app
COPY docker/location_refinement_stage/process_location_refinement.py /app
COPY docker/location_refinement_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
