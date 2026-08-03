"""Generate PoseText-style VQA from a single image's body keypoints.

Two entry points:

  * ``generate_from_molmo_output`` — pure-Python: take Molmo's ``<point>`` text
    (e.g. captured from a prior run or another model) and emit instruction-tuning
    pairs. No GPU / model install required.
  * ``generate_from_image`` — GPU path: load Molmo via
    ``vqasynth.pose.MolmoPoseLocalizer`` to predict the keypoints first.

Run as a script to print a few sample pairs built from a captured Molmo output:

    python examples/pose_estimation.py

This produces the same kind of (question, answer) data used to fine-tune Molmo
for body keypoint estimation — see issue #31 and salma-remyx/PoseText.
"""
from __future__ import annotations

from vqasynth.pose import build_pose_qa_pairs, parse_pose


# Captured Molmo <point> output for one person (normalized 0-100 coordinates),
# used by the CPU demo path so the script is runnable without a GPU.
SAMPLE_MOLMO_OUTPUT = (
    '<point x="50" y="12" alt="nose">'
    '<point x="46" y="10" alt="left eye">'
    '<point x="54" y="10" alt="right eye">'
    '<point x="40" y="22" alt="left shoulder">'
    '<point x="60" y="22" alt="right shoulder">'
    '<point x="32" y="40" alt="left elbow">'
    '<point x="68" y="40" alt="right elbow">'
    '<point x="28" y="55" alt="left wrist">'
    '<point x="72" y="55" alt="right wrist">'
    '<point x="44" y="48" alt="left hip">'
    '<point x="56" y="48" alt="right hip">'
    '<point x="42" y="70" alt="left knee">'
    '<point x="58" y="70" alt="right knee">'
    '<point x="41" y="90" alt="left ankle">'
    '<point x="59" y="90" alt="right ankle">'
)
SAMPLE_IMAGE_SIZE = (512, 512)


def generate_from_molmo_output(molmo_output, image_w, image_h, max_questions=8, seed=0):
    """Parse Molmo ``<point>`` text into a pose and build PoseText-style QA."""
    pose = parse_pose(molmo_output, image_w, image_h)
    return pose, build_pose_qa_pairs(pose, max_questions=max_questions, seed=seed)


def generate_from_image(image, model_name="cyan2k/molmo-7B-O-bnb-4bit",
                        max_questions=8, seed=0):
    """GPU path: predict keypoints with Molmo, then build QA pairs."""
    from vqasynth.pose import MolmoPoseLocalizer
    pose = MolmoPoseLocalizer(model_name=model_name).run(image)
    return pose, build_pose_qa_pairs(pose, max_questions=max_questions, seed=seed)


def _format(qa):
    lines = []
    for i, pair in enumerate(qa, 1):
        lines.append(f"Q{i}: {pair['question']}\nA{i}: {pair['answer']}")
    return "\n".join(lines)


if __name__ == "__main__":
    pose, qa = generate_from_molmo_output(
        SAMPLE_MOLMO_OUTPUT, SAMPLE_IMAGE_SIZE[0], SAMPLE_IMAGE_SIZE[1]
    )
    print(f"Detected {pose['num_detected']}/{len(pose['keypoints'])} body keypoints.\n")
    print(_format(qa))
