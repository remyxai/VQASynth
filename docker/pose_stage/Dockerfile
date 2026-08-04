FROM vqasynth:base
WORKDIR /app

# MediaPipe Pose is the default CPU-friendly keypoint backend (Apache-2.0).
# It runs on CPU, so this stage does not hard-depend on CUDA even though it
# builds on the CUDA base image for consistency with the other stages.
RUN pip install mediapipe

COPY docker/pose_stage/process_pose.py /app
COPY docker/pose_stage/entrypoint.sh /app
COPY config/config.yaml /app/config/config.yaml

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
