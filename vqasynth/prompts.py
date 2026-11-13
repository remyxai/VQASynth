"""Metric-aware spatial QA generation.

Distance and size predicates consume the per-object point clouds written by
``vqasynth.scene_fusion``, which are the pinhole unprojection of VGGT's depth
head. That head is only defined up to scene scale (``expm1`` activation, no
metric normalization), so raw cloud units are NOT meters — historically hidden
by a hard-coded ``* 10`` inside ``human_like_distance``.

This module replaces that fudge with per-scene metric calibration: a scale
factor mapping cloud units to real-world meters, recovered either by
unprojecting metric depth through the captured intrinsics
(:func:`unproject_depth_map`) or by anchoring the relative cloud against a
metric depth source such as DepthPro (:func:`calibrate_metric_scale`; the
repo's metric convention from ``vqasynth.depth`` is uint16 millimeters /
PIL ``I;16``). The scale threads through ``PromptGenerator`` so distance and
size predicates emit meters, and ``human_like_distance`` only formats.
"""
import re
import json
import math
import random
import numpy as np
from itertools import combinations
from vqasynth.prompt_templates import *


def depth_map_to_meters(depth_map):
    """
    Coerce a depth-map value to a float array in meters.

    Accepts the encodings this repo writes: float arrays are assumed to
    already be in meters; uint16 arrays and PIL ``I;16`` images (the
    ``vqasynth.depth`` DepthPro convention) are millimeter-encoded and get
    divided by 1000.

    Args:
        depth_map: array-like (H, W) depth, or a PIL depth image.

    Returns:
        np.ndarray of shape (H, W), dtype float64, in meters.
    """
    depth = np.asarray(depth_map)
    if depth.dtype == np.uint16:
        return depth.astype(np.float64) / 1000.0
    return depth.astype(np.float64)


def unproject_depth_map(depth_map, focal_length):
    """
    Unproject a depth map through pinhole intrinsics to camera-frame points,
    in the depth map's own units (meters in, meters out).

    Uses the OpenCV convention (x right, y down, z forward) with the principal
    point at the image center — the convention of the intrinsics VGGT predicts
    and ``vqasynth.scene_fusion`` unprojects with. Pixels with non-finite or
    non-positive depth are dropped.

    Args:
        depth_map: array-like (H, W) depth in any single unit.
        focal_length: focal length in pixels, w.r.t. the depth map's grid.

    Returns:
        np.ndarray of shape (N, 3) points.
    """
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"expected a 2D depth map, got shape {depth.shape}")
    focal_length = float(focal_length)
    if not math.isfinite(focal_length) or focal_length <= 0:
        raise ValueError(
            f"focal_length must be a positive pixel value, got {focal_length}"
        )

    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    x = (us[valid] - cx) * z / focal_length
    y = (vs[valid] - cy) * z / focal_length
    return np.stack([x, y, z], axis=1)


def calibrate_metric_scale(relative_depth, metric_depth):
    """
    Per-scene scale factor that maps a relative (up-to-scale) depth map onto a
    metric one.

    VGGT's depth head is scale-ambiguous, so its clouds need an external
    metric anchor; the repo's is DepthPro (``vqasynth.depth``). The scale is
    the median per-pixel ratio metric/relative — robust to the outliers and
    local errors both maps carry, and exact when the two maps differ by a
    single scalar (the standard relative-to-metric depth alignment).

    Args:
        relative_depth: (H, W) up-to-scale depth, e.g. VGGT's depth map.
        metric_depth: (H, W) metric depth in meters, e.g. DepthPro's, in any
            encoding :func:`depth_map_to_meters` accepts.

    Returns:
        float: multiply relative-unit distances by this to get meters.
    """
    relative = depth_map_to_meters(relative_depth)
    metric = depth_map_to_meters(metric_depth)
    if relative.shape != metric.shape:
        raise ValueError(
            "depth maps must share a grid to align, got "
            f"{relative.shape} vs {metric.shape}"
        )
    valid = (
        np.isfinite(relative) & (relative > 0)
        & np.isfinite(metric) & (metric > 0)
    )
    if not valid.any():
        raise ValueError("no valid overlapping pixels to calibrate metric scale")
    return float(np.median(metric[valid] / relative[valid]))


def _validate_metric_scale(metric_scale):
    scale = float(metric_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"metric_scale must be finite and positive, got {metric_scale}"
        )
    return scale


class PromptGenerator:
    def __init__(self, metric_scale=1.0):
        """
        Args:
            metric_scale (float): meters per point-cloud unit for the scenes
                this generator processes. 1.0 means the clouds are already
                metric (e.g. unprojected from DepthPro metric depth with
                ``unproject_depth_map``); otherwise recover it per scene with
                ``calibrate_metric_scale``. A per-example ``metric_scale``
                dataset column takes precedence (see ``apply_transform``).
        """
        self.default_metric_scale = _validate_metric_scale(metric_scale)
        self.metric_scale = self.default_metric_scale

    def _metric(self, distance_in_cloud_units):
        """Convert a raw point-cloud distance to meters via the active scale."""
        return float(distance_in_cloud_units) * self.metric_scale

    def human_like_distance(self, distance_meters):
        """
        Converts a distance in meters to a more human-readable format (e.g., centimeters, feet).

        The input must already be calibrated to real-world meters (see
        ``_metric`` / ``calibrate_metric_scale``); no unit fudge is applied
        here.

        Args:
            distance_meters (float): The distance in meters.

        Returns:
            str: A string representation of the human-readable distance with units.
        """
        if distance_meters < 1:
            choices = [
                (
                    round(distance_meters * 100, 2),
                    "centimeters",
                    0.6,
                ),  # Centimeters for very small distances
                (
                    round(distance_meters * 39.37, 2),
                    "inches",
                    0.4,
                ),  # Inches for small distances
            ]
        elif distance_meters < 3:
            choices = [
                (round(distance_meters, 2), "meters", 0.6),  # Meters for clarity
                (round(distance_meters * 3.281, 2), "feet", 0.4),
            ]
        else:
            choices = [
                (
                    round(distance_meters, 2),
                    "meters",
                    0.7,
                ),  # Meters for larger distances
                (
                    round(distance_meters * 3.281, 2),
                    "feet",
                    0.3,
                ),  # Feet for additional context
            ]

        # Normalize probabilities and make a selection
        total_probability = sum(prob for _, _, prob in choices)
        cumulative_distribution = []
        cumulative_sum = 0
        for value, unit, probability in choices:
            cumulative_sum += probability / total_probability  # Normalize probabilities
            cumulative_distribution.append((cumulative_sum, value, unit))

        # Randomly choose based on the cumulative distribution
        r = random.random()
        for cumulative_prob, value, unit in cumulative_distribution:
            if r < cumulative_prob:
                if math.isnan(value) or value == 0.0:
                    return f"{value} {unit}"

                rounding_precision = random.choice([0, 1, 2])
                rounded_value = round(value, rounding_precision)
                return f"{rounded_value} {unit}"

        # Fallback to the last choice if something goes wrong
        final_value = choices[-1][0]
        final_choice_unit = choices[-1][1]

        if math.isnan(final_value) or final_value == 0.0:
            return f"{final_value} {final_choice_unit}"

        rounding_precision = random.choice([0, 1, 2])
        final_choice_value = round(final_value, rounding_precision)
        return f"{final_choice_value} {final_choice_unit}"

    def extract_distance_from_result(self, result):
        """
        Extracts the distance value from the result string in the form 'question + " Answer: " + answer'.
        The answer part may contain descriptive text and the distance. We extract the numeric value from the answer.
        """
        # Split the result string into the question and answer parts
        question_part, answer_part = result.split(" Answer: ", 1)

        # Use regex to find the first occurrence of a floating point or integer number in the answer
        match = re.search(r"(\d+\.\d+|\d+)", answer_part)
        if match:
            # Convert the matched numeric part to float and return
            distance = float(match.group(0))
            return distance
        else:
            # If no numeric part is found, return None
            return None

    def is_valid_result(self, result):
        """
        Checks if a result string contains a valid, non-NaN, non-zero distance.
        """
        distance = self.extract_distance_from_result(result)

        # Return False if distance is None or is NaN
        if distance is None or math.isnan(distance):
            return False

        # Filter exact zero distances
        if distance == 0.0:
            return False

        return True

    def left_predicate(self, A, B):
        template_questions = left_predicate_questions
        true_responses = left_true_responses
        false_responses = left_false_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_pos = A_cloud.get_center()
        B_pos = B_cloud.get_center()

        is_left = A_pos[0] < B_pos[0]  # Compare X coordinates

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_left else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def right_predicate(self, A, B):
        template_questions = right_predicate_questions
        true_responses = right_true_responses
        false_responses = right_false_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_pos = A_cloud.get_center()
        B_pos = B_cloud.get_center()

        is_right = A_pos[0] > B_pos[0]  # Compare X coordinates

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_right else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def above_predicate(self, A, B):
        template_questions = above_predicate_questions
        true_responses = above_true_responses
        false_responses = above_false_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_pos = A_cloud.get_center()
        B_pos = B_cloud.get_center()

        is_above = A_pos[1] > B_pos[1]

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_above else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def below_predicate(self, A, B):
        template_questions = below_predicate_questions
        true_responses = below_true_responses
        false_responses = below_false_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_pos = A_cloud.get_center()
        B_pos = B_cloud.get_center()

        is_below = A_pos[1] < B_pos[1]

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_below else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def wide_predicate(self, A, B):
        template_questions = wide_predicate_questions
        true_responses = wide_true_responses
        false_responses = wide_false_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        width_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[0]
        width_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[0]

        is_wider = width_A > width_B

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_wider else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def big_predicate(self, A, B):
        template_questions = big_predicate_questions
        true_responses = big_true_responses
        false_responses = big_false_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        extent_A = A_cloud.get_axis_aligned_bounding_box().get_extent()
        volume_A = extent_A[0] * extent_A[1] * extent_A[2]

        extent_B = B_cloud.get_axis_aligned_bounding_box().get_extent()
        volume_B = extent_B[0] * extent_B[1] * extent_B[2]

        is_bigger = volume_A > volume_B

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_bigger else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def tall_predicate(self, A, B):
        template_questions = tall_predicate_questions
        true_responses = tall_true_responses
        false_responses = tall_false_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
        height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]

        is_taller = height_A > height_B

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_taller else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def short_predicate(self, A, B):
        template_questions = short_predicate_questions
        true_responses = short_true_responses
        false_responses = short_false_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
        height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]

        is_shorter = height_A < height_B

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_shorter else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def thin_predicate(self, A, B):
        template_questions = thin_predicate_questions
        true_responses = thin_true_responses
        false_responses = thin_false_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        width_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[0]
        width_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[0]

        is_thinner = width_A < width_B

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_thinner else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def small_predicate(self, A, B):
        template_questions = small_predicate_questions
        true_responses = small_true_responses
        false_responses = small_false_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        extent_A = A_cloud.get_axis_aligned_bounding_box().get_extent()
        volume_A = extent_A[0] * extent_A[1] * extent_A[2]

        extent_B = B_cloud.get_axis_aligned_bounding_box().get_extent()
        volume_B = extent_B[0] * extent_B[1] * extent_B[2]

        is_smaller = volume_A < volume_B

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_smaller else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def behind_predicate(self, A, B):
        template_questions = behind_predicate_questions
        true_responses = behind_true
        false_responses = behind_false

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_center = A_cloud.get_axis_aligned_bounding_box().get_center()
        B_center = B_cloud.get_axis_aligned_bounding_box().get_center()
        is_behind = A_center[2] > B_center[2]

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_behind else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    def front_predicate(self, A, B):
        template_questions = front_predicate_questions
        true_responses = front_true
        false_responses = front_false

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_center = A_cloud.get_axis_aligned_bounding_box().get_center()
        B_center = B_cloud.get_axis_aligned_bounding_box().get_center()
        is_in_front = A_center[2] < B_center[2]

        question_template = random.choice(template_questions)
        response_template = random.choice(
            true_responses if is_in_front else false_responses
        )

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

        return question + " Answer: " + answer

    # Choice prompts

    def left_choice(self, A, B):
        template_questions = left_choice_questions
        template_responses = left_choice_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_pos = A_cloud.get_center()
        B_pos = B_cloud.get_center()

        more_left = A_desc if A_pos[0] < B_pos[0] else B_desc

        question_template = random.choice(template_questions)
        answer_template = random.choice(template_responses)

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = answer_template.replace("[X]", more_left)

        return question + " Answer: " + answer

    def right_choice(self, A, B):
        template_questions = right_choice_questions
        template_responses = right_choice_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_pos = A_cloud.get_center()
        B_pos = B_cloud.get_center()

        more_right = A_desc if A_pos[0] > B_pos[0] else B_desc

        question_template = random.choice(template_questions)
        answer_template = random.choice(template_responses)

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = answer_template.replace("[X]", more_right)

        return question + " Answer: " + answer

    def above_choice(self, A, B):
        template_questions = above_choice_questions
        template_responses = above_choice_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_pos = A_cloud.get_center()
        B_pos = B_cloud.get_center()

        more_above = A_desc if A_pos[1] > B_pos[1] else B_desc

        question_template = random.choice(template_questions)
        answer_template = random.choice(template_responses)

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = answer_template.replace("[X]", more_above)

        return question + " Answer: " + answer

    def below_choice(self, A, B):
        template_questions = below_choice_questions
        template_responses = below_choice_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_pos = A_cloud.get_center()
        B_pos = B_cloud.get_center()

        more_below = A_desc if A_pos[1] < B_pos[1] else B_desc

        question_template = random.choice(template_questions)
        answer_template = random.choice(template_responses)

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = answer_template.replace("[X]", more_below)

        return question + " Answer: " + answer

    def tall_choice(self, A, B):
        template_questions = tall_choice_questions
        template_responses = tall_choice_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
        height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]

        taller = A_desc if height_A > height_B else B_desc

        question_template = random.choice(template_questions)
        answer_template = random.choice(template_responses)

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = answer_template.replace("[X]", taller)

        return question + " Answer: " + answer

    def short_choice(self, A, B):
        template_questions = short_choice_questions
        template_responses = short_choice_responses

        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
        height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]

        shorter = A_desc if height_A < height_B else B_desc

        question_template = random.choice(template_questions)
        answer_template = random.choice(template_responses)

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = answer_template.replace("[X]", shorter)

        return question + " Answer: " + answer

    # Distance prompts

    def generate_spatial_reasoning_data(
        self, A, B, human_readable_dist, template_questions, template_answers
    ):
        A_desc, B_desc = A[0].lower(), B[0].lower()

        question_template = random.choice(template_questions)
        answer_template = random.choice(template_answers)

        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = (
            answer_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[X]", human_readable_dist)
        )

        return question + " Answer: " + answer

    def vertical_distance_data(self, A, B):
        template_questions = vertical_distance_questions
        template_answers = vertical_distance_answers

        A_center = A[1].get_axis_aligned_bounding_box().get_center()
        B_center = B[1].get_axis_aligned_bounding_box().get_center()
        vertical_distance = abs(A_center[1] - B_center[1])
        human_readable_dist = self.human_like_distance(
            self._metric(vertical_distance)
        )

        return self.generate_spatial_reasoning_data(
            A, B, human_readable_dist, template_questions, template_answers
        )

    def horizontal_distance_data(self, A, B):
        template_questions = horizontal_distance_questions
        template_answers = horizontal_distance_answers

        A_center = A[1].get_axis_aligned_bounding_box().get_center()
        B_center = B[1].get_axis_aligned_bounding_box().get_center()
        horizontal_distance = np.sqrt(
            (A_center[0] - B_center[0]) ** 2 + (A_center[2] - B_center[2]) ** 2
        )

        human_readable_dist = self.human_like_distance(
            self._metric(horizontal_distance)
        )
        return self.generate_spatial_reasoning_data(
            A, B, human_readable_dist, template_questions, template_answers
        )

    def width_data(self, A, B=None):
        A_desc = A[0].lower()

        template_questions = width_questions
        template_answers = width_answers

        width = A[1].get_axis_aligned_bounding_box().get_extent()[0]

        human_readable_width = self.human_like_distance(self._metric(width))
        question_template = random.choice(template_questions)
        answer_template = random.choice(template_answers)

        question = question_template.replace("[A]", A_desc)
        answer = answer_template.replace("[A]", A_desc).replace(
            "[X]", human_readable_width
        )

        return question + " Answer: " + answer

    def height_data(self, A, B=None):
        A_desc = A[0].lower()

        template_questions = height_questions
        template_answers = height_answers

        height = A[1].get_axis_aligned_bounding_box().get_extent()[1]

        human_readable_height = self.human_like_distance(self._metric(height))
        question_template = random.choice(template_questions)
        answer_template = random.choice(template_answers)

        question = question_template.replace("[A]", A_desc)
        answer = answer_template.replace("[A]", A_desc).replace(
            "[X]", human_readable_height
        )

        return question + " Answer: " + answer

    # Perspective-taking / cross-view prompts (multi-view scenes only)
    #
    # These families consume the per-frame camera poses surfaced alongside the
    # fused multi-view point cloud. Expected input: an ``extrinsics`` entry of
    # shape (S, 3, 4) holding camera-from-world poses [R | t] in the OpenCV
    # convention (x right, y down, z forward) — the same convention VGGT's
    # ``pose_encoding_to_extri_intri`` emits — with each pose expressed in the
    # fused cloud's (world) frame, so p_camera = R @ p_world + t. They are
    # gated on >= 2 poses; single-frame scenes never see them.

    def parse_extrinsics(self, extrinsics, min_frames=2):
        """
        Normalize per-frame camera extrinsics into an (S, 3, 4) numpy array of
        camera-from-world poses, or None when the input does not describe at
        least ``min_frames`` usable poses.

        Accepts nested lists, numpy arrays, or torch tensors of shape
        (S, 3, 4) / (S, 4, 4), or a single (3, 4) / (4, 4) pose. Anything
        else (including None, non-finite values, or rotation-only shapes)
        yields None so callers can fall back to single-frame behavior.
        """
        if extrinsics is None:
            return None
        try:
            if hasattr(extrinsics, "detach"):  # torch tensor, maybe on GPU
                extrinsics = extrinsics.detach().cpu().numpy()
            poses = np.asarray(extrinsics, dtype=np.float64)
        except Exception:
            return None

        if poses.ndim == 2:  # a single (3, 4) or (4, 4) pose
            poses = poses[None, ...]
        if poses.ndim != 3 or poses.shape[0] < min_frames:
            return None
        if poses.shape[1:] == (4, 4):
            poses = poses[:, :3, :]
        if poses.shape[1:] != (3, 4):
            return None
        if not np.all(np.isfinite(poses)):
            return None
        return poses

    @staticmethod
    def transform_points_to_camera(points_world, extrinsic):
        """
        Map world-frame points into a camera's frame using its camera-from-world
        pose: p_camera = R @ p_world + t.

        Args:
            points_world ((N, 3) or (3,) array-like): points in the fused
                cloud's (world) frame.
            extrinsic ((3, 4) or (4, 4) array-like): camera-from-world pose.

        Returns:
            np.ndarray of the same leading shape in camera coordinates.
        """
        pose = np.asarray(extrinsic, dtype=np.float64)[:3, :]
        pts = np.asarray(points_world, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        pts_cam = pts @ pose[:, :3].T + pose[:, 3]
        return pts_cam[0] if single else pts_cam

    @staticmethod
    def _viewpoint_label(frame_index):
        """Human-facing 1-indexed name for a frame, e.g. ``"camera 2"``."""
        return f"camera {frame_index + 1}"

    def _perspective_relation(self, A, B, frame, extrinsics):
        """
        Express both object centers in ``frame``'s camera coordinates.

        Returns:
            (A_pos, B_pos) camera-frame center arrays, or None when the
            extrinsics/frame do not describe a usable pose.
        """
        poses = self.parse_extrinsics(extrinsics, min_frames=1)
        if poses is None or not 0 <= frame < len(poses):
            return None
        A_pos = self.transform_points_to_camera(A[1].get_center(), poses[frame])
        B_pos = self.transform_points_to_camera(B[1].get_center(), poses[frame])
        return A_pos, B_pos

    def perspective_left_predicate(self, A, B, frame, extrinsics):
        A_desc, B_desc = A[0].lower(), B[0].lower()
        relation = self._perspective_relation(A, B, frame, extrinsics)
        if relation is None:
            raise ValueError("perspective predicates need a usable per-frame extrinsic")
        A_pos, B_pos = relation

        is_left = A_pos[0] < B_pos[0]  # camera-frame x: right in OpenCV poses

        viewpoint = self._viewpoint_label(frame)
        question_template = random.choice(perspective_left_predicate_questions)
        response_template = random.choice(
            perspective_left_true_responses if is_left else perspective_left_false_responses
        )

        question = (
            question_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", viewpoint)
        )
        answer = (
            response_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", viewpoint)
        )

        return question + " Answer: " + answer

    def perspective_right_predicate(self, A, B, frame, extrinsics):
        A_desc, B_desc = A[0].lower(), B[0].lower()
        relation = self._perspective_relation(A, B, frame, extrinsics)
        if relation is None:
            raise ValueError("perspective predicates need a usable per-frame extrinsic")
        A_pos, B_pos = relation

        is_right = A_pos[0] > B_pos[0]

        viewpoint = self._viewpoint_label(frame)
        question_template = random.choice(perspective_right_predicate_questions)
        response_template = random.choice(
            perspective_right_true_responses if is_right else perspective_right_false_responses
        )

        question = (
            question_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", viewpoint)
        )
        answer = (
            response_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", viewpoint)
        )

        return question + " Answer: " + answer

    def perspective_front_predicate(self, A, B, frame, extrinsics):
        A_desc, B_desc = A[0].lower(), B[0].lower()
        relation = self._perspective_relation(A, B, frame, extrinsics)
        if relation is None:
            raise ValueError("perspective predicates need a usable per-frame extrinsic")
        A_pos, B_pos = relation

        is_in_front = A_pos[2] < B_pos[2]  # camera-frame depth: smaller z is closer

        viewpoint = self._viewpoint_label(frame)
        question_template = random.choice(perspective_front_predicate_questions)
        response_template = random.choice(
            perspective_front_true_responses if is_in_front else perspective_front_false_responses
        )

        question = (
            question_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", viewpoint)
        )
        answer = (
            response_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", viewpoint)
        )

        return question + " Answer: " + answer

    def perspective_behind_predicate(self, A, B, frame, extrinsics):
        A_desc, B_desc = A[0].lower(), B[0].lower()
        relation = self._perspective_relation(A, B, frame, extrinsics)
        if relation is None:
            raise ValueError("perspective predicates need a usable per-frame extrinsic")
        A_pos, B_pos = relation

        is_behind = A_pos[2] > B_pos[2]

        viewpoint = self._viewpoint_label(frame)
        question_template = random.choice(perspective_behind_predicate_questions)
        response_template = random.choice(
            perspective_behind_true_responses if is_behind else perspective_behind_false_responses
        )

        question = (
            question_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", viewpoint)
        )
        answer = (
            response_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", viewpoint)
        )

        return question + " Answer: " + answer

    def perspective_left_choice(self, A, B, frame, extrinsics):
        A_desc, B_desc = A[0].lower(), B[0].lower()
        relation = self._perspective_relation(A, B, frame, extrinsics)
        if relation is None:
            raise ValueError("perspective predicates need a usable per-frame extrinsic")
        A_pos, B_pos = relation

        more_left = A_desc if A_pos[0] < B_pos[0] else B_desc

        viewpoint = self._viewpoint_label(frame)
        question_template = random.choice(perspective_left_choice_questions)
        answer_template = random.choice(perspective_left_choice_responses)

        question = (
            question_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", viewpoint)
        )
        answer = answer_template.replace("[V]", viewpoint).replace("[X]", more_left)

        return question + " Answer: " + answer

    def cross_view_order_predicate(self, A, B, frame_v, frame_w, extrinsics):
        """
        Ask whether the left-right order of the pair is the same in two
        reference frames.

        Args:
            frame_v: index of the egocentric viewpoint's pose in ``extrinsics``.
            frame_w: index of a second viewpoint's pose, or None to compare
                against the allocentric world (fused-cloud) layout.
        """
        A_desc, B_desc = A[0].lower(), B[0].lower()
        relation_v = self._perspective_relation(A, B, frame_v, extrinsics)
        if relation_v is None:
            raise ValueError("cross-view predicates need a usable per-frame extrinsic")
        A_v, B_v = relation_v
        v_left_of = A_v[0] < B_v[0]

        if frame_w is None:
            A_w, B_w = A[1].get_center(), B[1].get_center()
            w_label = "scene's overall layout"
        else:
            relation_w = self._perspective_relation(A, B, frame_w, extrinsics)
            if relation_w is None:
                raise ValueError("cross-view predicates need a usable per-frame extrinsic")
            A_w, B_w = relation_w
            w_label = f"{self._viewpoint_label(frame_w)} view"

        w_left_of = A_w[0] < B_w[0]
        orders_agree = v_left_of == w_left_of

        v_label = f"{self._viewpoint_label(frame_v)} view"
        question_template = random.choice(cross_view_order_questions)
        response_template = random.choice(
            cross_view_order_true_responses
            if orders_agree
            else cross_view_order_false_responses
        )

        question = (
            question_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", v_label)
            .replace("[W]", w_label)
        )
        answer = (
            response_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[V]", v_label)
            .replace("[W]", w_label)
        )

        return question + " Answer: " + answer

    def build_viewpoint_variants(self, extrinsics):
        """
        Bind the perspective-taking / cross-view families to concrete
        viewpoints for this scene: one egocentric frame plus a second
        reference frame (another camera when the scene has one, otherwise the
        allocentric world layout).

        Returns:
            List of (A, B) -> str callables ready to drop into
            ``evaluate_predicates_on_pairs``, or [] when ``extrinsics`` does
            not describe at least two camera poses (single-frame back-compat).
        """
        poses = self.parse_extrinsics(extrinsics, min_frames=2)
        if poses is None:
            return []

        frame = random.randrange(len(poses))
        if len(poses) > 2:
            frame_w = random.choice([i for i in range(len(poses)) if i != frame])
        else:
            frame_w = None  # egocentric vs allocentric (world layout)

        return [
            lambda A, B: self.perspective_left_predicate(A, B, frame, poses),
            lambda A, B: self.perspective_right_predicate(A, B, frame, poses),
            lambda A, B: self.perspective_front_predicate(A, B, frame, poses),
            lambda A, B: self.perspective_behind_predicate(A, B, frame, poses),
            lambda A, B: self.perspective_left_choice(A, B, frame, poses),
            lambda A, B: self.cross_view_order_predicate(A, B, frame, frame_w, poses),
        ]

    def evaluate_predicates_on_pairs(self, pairs, is_canonicalized, extrinsics=None):
        """
        Evaluate predicates on object pairs, ensuring a balance between distance-related predicates
        and other types of predicates. If invalid distance results are found, they are replaced with
        other types of predicates.

        Args:
            pairs: List of object pairs to evaluate.
            is_canonicalized: Boolean indicating if canonicalized predicates should be included.
            extrinsics: Optional per-frame camera poses (S, 3, 4), camera-from-world,
                aligned with the fused cloud's frame. When at least two poses are
                present the perspective-taking / cross-view families join the
                candidate pool; otherwise they are skipped (single-frame back-compat).

        Returns:
            List of results for each object pair.
        """
        all_prompt_variants = [
            self.left_predicate,
            self.right_predicate,
            self.wide_predicate,
            self.big_predicate,
            self.thin_predicate,
            self.small_predicate,
            self.behind_predicate,
            self.front_predicate,
            self.left_choice,
            self.right_choice,
        ]

        add_canonicalized = [
            self.tall_choice,
            self.above_predicate,
            self.below_predicate,
            self.short_choice,
            self.below_choice,
            self.tall_predicate,
            self.short_predicate,
            self.above_choice,
            self.vertical_distance_data,
            self.horizontal_distance_data,
            self.width_data,
            self.height_data,
        ]

        if is_canonicalized:
            all_prompt_variants += add_canonicalized

        # Perspective-taking / cross-view families only join the pool when the
        # fused scene carries >= 2 registered camera poses; single-frame scenes
        # (no extrinsics, or a single pose) skip them entirely.
        all_prompt_variants += self.build_viewpoint_variants(extrinsics)

        # Distance-related predicates
        distance_related_predicates = [
            self.vertical_distance_data,
            self.horizontal_distance_data,
            self.width_data,
            self.height_data,
        ]

        selected_distance_predicates = distance_related_predicates
        selected_other_predicates = random.sample(all_prompt_variants, k=2)
        selected_predicates_choices = (
            selected_distance_predicates + selected_other_predicates
        )

        results = []
        filtered_out_count = 0

        for A, B in pairs:
            pair_results = []
            for prompt_func in selected_predicates_choices:
                pair_results.append(prompt_func(A, B))

            distance = np.asarray(A[1].compute_point_cloud_distance(B[1])).mean()
            distance_str = self.human_like_distance(self._metric(distance))

            for _ in range(5):
                reasoning_data = self.generate_spatial_reasoning_data(
                    A,
                    B,
                    distance_str,
                    distance_template_questions,
                    distance_template_answers,
                )
                pair_results.append(reasoning_data)

            results.extend(pair_results)

        # Filter out invalid results (nan or 0.0 distances)
        valid_results = [result for result in results if self.is_valid_result(result)]
        filtered_out_count = len(results) - len(valid_results)

        # Replace them with non-distance-measuring prompts
        if filtered_out_count > 0:
            replacements = []
            for A, B in pairs:
                replacement_prompts = random.choices(
                    selected_other_predicates, k=filtered_out_count
                )
                for prompt_func in replacement_prompts:
                    replacements.append(prompt_func(A, B))

            valid_results.extend(replacements)

        valid_results = list(set(valid_results))
        random.shuffle(valid_results)

        return valid_results

    def _resolve_metric_scale(self, example):
        """
        Per-example metric scale: the ``metric_scale`` column when present
        (emitted as a per-row list by scene-fusion-style stages), else this
        generator's default. Lets a calibration stage anchor each scene
        without any change to the prompt stage.
        """
        value = example.get("metric_scale")
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(
                    f"expected one metric_scale per example, got {len(value)}"
                )
            value = value[0]
        if value is None:
            return self.default_metric_scale
        return _validate_metric_scale(value)

    def run(self, captions, pointclouds, is_canonicalized, extrinsics=None, metric_scale=None):
        # restore_pointclouds (i.e. vqasynth.scene_fusion) pulls in the
        # VGGT/CUDA stack; importing it here keeps this module usable — and
        # unit-testable — without it.
        from vqasynth.scene_fusion import restore_pointclouds

        if metric_scale is not None:
            self.metric_scale = _validate_metric_scale(metric_scale)
        else:
            self.metric_scale = self.default_metric_scale
        pointclouds = restore_pointclouds(pointclouds)
        try:
            objects = list(zip(captions, pointclouds))
            all_pairs = [
                (i, j)
                for i in range(len(objects))
                for j in range(len(objects))
                if i != j
            ]
            random.shuffle(all_pairs)
            selected_pairs = all_pairs[:5]
            object_pairs = [(objects[i], objects[j]) for i, j in selected_pairs]
            prompts = self.evaluate_predicates_on_pairs(
                object_pairs, is_canonicalized, extrinsics=extrinsics
            )
        except:
            prompts = []
        return prompts

    def create_messages_from_prompts(self, prompts):
        """
        Converts a list of prompts into structured messages for the dataset.

        Args:
            prompts (list): A list of prompt strings.

        Returns:
            A list of message dictionaries formatted for user and assistant roles.
        """
        messages = []
        first_prompt = True

        for prompt in prompts:
            if "Answer: " in prompt:
                question, answer = prompt.split("Answer: ", 1)

                # Add user message (include image only for the first prompt)
                content = [{"index": None, "text": question.strip(), "type": "text"}]
                if first_prompt:
                    content.insert(0, {"index": 0, "text": None, "type": "image"})

                messages.append({"content": content, "role": "user"})

                messages.append(
                    {
                        "content": [
                            {"index": None, "text": answer.strip(), "type": "text"}
                        ],
                        "role": "assistant",
                    }
                )

                first_prompt = False

        return messages

    @staticmethod
    def _scene_scalar(value):
        """Reduce a surfaced confidence column to one scene-level scalar.

        Accepts a scalar (returned as-is), a list/tuple (mean of its finite
        entries), or None/garbage (None). Used for scene/frame-level signals like
        the cross-view ``inlier_ratio`` that apply to the registration as a whole
        rather than to a single object.
        """
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            vals = []
            for v in value:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            return round(float(np.mean(vals)), 3) if vals else None
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None

    def build_grounded_state(
        self,
        captions,
        pointclouds,
        extrinsics=None,
        depth_conf=None,
        inlier_ratio=None,
        max_pairs=5,
    ):
        """Assemble the compact grounded-scene-state record consumed by
        :class:`~vqasynth.r1_reasoning.SoftTraceReasoner`.

        Per object: label + intrinsics-calibrated metric center (Phase 1a) +
        surfaced per-object depth confidence (Phase 1b). Per sampled pair: the
        calibrated center-to-center distance in meters, carrying the scene's
        cross-view match reliability (Phase 2c ``inlier_ratio``). Plus the frame
        count (Phase 2a multi-view). Returned as a JSON string so it is a robust
        free-form column across the dataset.

        Best-effort by design: any failure yields ``"{}"`` so the teacher falls
        back to ungrounded tracing (back-compat with single-view / sparse scenes).
        """
        # scene_fusion pulls in the VGGT/CUDA stack; import lazily (as run()
        # does) so this module stays importable/testable without it.
        try:
            from vqasynth.scene_fusion import restore_pointclouds

            clouds = restore_pointclouds(pointclouds)
        except Exception:
            return "{}"
        try:
            n = min(len(captions), len(clouds))
            scene_ir = self._scene_scalar(inlier_ratio)

            objects, centers = [], []
            for idx in range(n):
                try:
                    center = [round(float(c), 3) for c in clouds[idx].get_center()]
                except Exception:
                    center = None
                centers.append(center)
                obj = {"label": captions[idx], "center_m": center}
                dc = depth_conf[idx] if isinstance(depth_conf, (list, tuple)) and idx < len(depth_conf) else None
                dc = self._scene_scalar(dc) if dc is not None else None
                if dc is not None:
                    obj["depth_conf"] = dc
                objects.append(obj)

            pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pair_indices)
            pairs = []
            for i, j in pair_indices[:max_pairs]:
                if centers[i] is None or centers[j] is None:
                    continue
                dist_m = float(np.linalg.norm(np.asarray(centers[i]) - np.asarray(centers[j])))
                pair = {"a": captions[i], "b": captions[j], "distance_m": round(dist_m, 3)}
                if scene_ir is not None:
                    pair["inlier_ratio"] = scene_ir
                pairs.append(pair)

            frames = len(extrinsics) if isinstance(extrinsics, (list, tuple)) and extrinsics else 1
            return json.dumps({"frames": frames, "objects": objects, "pairs": pairs})
        except Exception:
            return "{}"

    def apply_transform(self, example):
        """
        Process a single example to produce:
          - "prompts": list of strings (from self.run)
          - "truncated_prompts": list of strings (first 5 shuffled prompts)
          - "messages": a list of message dictionaries with the nested structure:

              Each message dictionary:
              {
                 "role": <"user" or "assistant">,
                 "content": [
                     { "index": 0 or None, "text": <string or None>, "type": <"image" or "text"> },
                     ... (possibly multiple items)
                 ]
              }

              Specifically, the first message always has:
                 "content": [
                     {"index": 0, "text": None, "type": "image"},
                     {"index": None, "text": <first prompt>, "type": "text"}
                 ]
              and subsequent messages have:
                 "content": [ { "index": None, "text": <prompt>, "type": "text" } ]

        If any error occurs for this sample, returns None (and the sample will be dropped).

        Distance/size values are metric: the per-example ``metric_scale``
        column takes precedence (see ``_resolve_metric_scale``), else the
        generator's default.
        """
        try:
            # Process a single sample. "extrinsics" is the optional per-frame
            # camera-pose column carried by multi-view (fused) scenes; absent
            # on single-frame scenes, which simply skip the viewpoint families.
            extrinsics = example.get("extrinsics") if isinstance(example, dict) else None
            prompts = self.run(
                example["captions"],
                example["pointclouds"],
                example["is_canonicalized"],
                extrinsics=extrinsics,
                metric_scale=self._resolve_metric_scale(example),
            )
            random.shuffle(prompts)
            truncated = prompts[:5]
            messages = self.create_messages_from_prompts(truncated)
            grounded_state = self.build_grounded_state(
                example["captions"],
                example["pointclouds"],
                extrinsics=extrinsics,
                depth_conf=example.get("depth_conf") if isinstance(example, dict) else None,
                inlier_ratio=example.get("inlier_ratio") if isinstance(example, dict) else None,
            )
            return {
                "prompts": prompts,
                "truncated_prompts": truncated,
                "messages": messages,
                "grounded_state": grounded_state,
            }
        except Exception as e:
            print(f"Error processing sample, skipping: {e}")
            return None

