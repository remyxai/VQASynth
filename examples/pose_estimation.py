"""Generate PoseText-style SFT samples from a person's body keypoints.

Data-generation direction (keypoint-source-first): keypoints come from a
lightweight pose model, and the emitter renders them as Molmo-style
``<point>`` answers. Molmo is the *target* of this distillation, not the
source of the keypoints. See issue #31 and salma-remyx/PoseText.

Two entry points:

  * ``generate_from_keypoints`` — pure-Python: take a keypoint set (e.g. from
    an annotated dataset like COCO/MPII, or captured from a pose model) and
    emit instruction-tuning samples. No GPU / model install required.
  * ``generate_from_image`` — runs a pose backend over a PIL image via
    ``vqasynth.pose.KeypointExtractor`` (default MediaPipe, CPU-friendly) to
    predict the keypoints first.

Run as a script to print a few sample SFT training pairs built from a
synthetic keypoint fixture:

    python examples/pose_estimation.py
"""
from __future__ import annotations

from vqasynth.pose import (
    KeypointExtractor,
    build_pose_messages,
    pose_from_keypoints,
)


# A synthetic 17-keypoint set for a 256x256 image, in COCO order
# (name -> [x_pixel, y_pixel]). Only the visible joints are listed; the
# emitter skips the rest. Used by the CPU demo path so the script runs
# without a pose-model install.
SAMPLE_IMAGE_SIZE = (256, 256)
SAMPLE_KEYPOINTS = {
    "nose": [128, 30],
    "left_shoulder": [80, 64],
    "right_shoulder": [176, 64],
    "left_elbow": [56, 110],
    "right_elbow": [200, 110],
    "left_wrist": [44, 150],
    "right_wrist": [212, 150],
    "left_hip": [96, 130],
    "right_hip": [160, 130],
    "left_knee": [100, 190],
    "right_knee": [156, 190],
    "left_ankle": [96, 240],
    "right_ankle": [160, 240],
}


def generate_from_keypoints(keypoints, image_size, max_questions=8, seed=0):
    """Build PoseText-style SFT chat samples from a keypoint set.

    ``keypoints`` follows the format accepted by
    :func:`vqasynth.pose.pose_from_keypoints` (a list of 17 ``(x, y)`` /
    ``None`` in COCO order, or a ``{name: (x, y)}`` mapping).
    """
    pose = pose_from_keypoints(keypoints, image_size)
    return pose, build_pose_messages(pose, max_questions=max_questions, seed=seed)


def generate_from_image(image, backend="mediapipe", max_questions=8, seed=0):
    """Run a pose backend over ``image`` and build SFT samples per person."""
    extractor = KeypointExtractor(backend=backend)
    pose_list = extractor.extract(image)
    samples = []
    for pose in pose_list:
        samples.extend(build_pose_messages(pose, max_questions=max_questions, seed=seed))
    return pose_list, samples


def _format(samples):
    lines = []
    for i, sample in enumerate(samples, 1):
        msgs = sample["messages"]
        question = next(c["text"] for c in msgs[0]["content"] if c["type"] == "text")
        answer = msgs[1]["content"][0]["text"]
        lines.append(f"Q{i}: {question}\nA{i}: {answer}")
    return "\n".join(lines)


if __name__ == "__main__":
    pose, samples = generate_from_keypoints(SAMPLE_KEYPOINTS, SAMPLE_IMAGE_SIZE)
    print(f"Detected {pose['num_detected']}/{len(pose['keypoints'])} body keypoints.\n")
    print(_format(samples))
