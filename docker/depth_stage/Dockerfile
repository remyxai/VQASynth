FROM vqasynth:base
WORKDIR /app

ENV PATH="/usr/local/cuda-11.8/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}"

RUN pip uninstall -y onnxruntime onnxruntime-gpu
RUN pip install onnxruntime-gpu==1.18.1

COPY docker/depth_stage/process_depth.py /app
COPY docker/depth_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
