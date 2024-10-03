FROM smellslikeml/vqasynth:base
WORKDIR /app

COPY process_segment.py /app
COPY entrypoint.sh /app

RUN pip install pandas
RUN chmod +x /app/entrypoint.sh
# install sam2
RUN python3 -m pip install 'git+https://github.com/facebookresearch/sam2.git'

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
