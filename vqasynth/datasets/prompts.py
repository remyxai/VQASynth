import random
import numpy as np
from itertools import combinations
from vqasynth.datasets.prompt_templates import *

from vqasynth.datasets.pointcloud import human_like_distance, calculate_distances_between_point_clouds


# Predicate prompts

def left_predicate(A, B):
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
    response_template = random.choice(true_responses if is_left else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer


def right_predicate(A, B):
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
    response_template = random.choice(true_responses if is_right else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer


def above_predicate(A, B):
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
    response_template = random.choice(true_responses if is_above else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer


def below_predicate(A, B):
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
    response_template = random.choice(true_responses if is_below else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer


def wide_predicate(A, B):
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
    response_template = random.choice(true_responses if is_wider else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer


def big_predicate(A, B):
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
    response_template = random.choice(true_responses if is_bigger else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer

def tall_predicate(A, B):
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
    response_template = random.choice(true_responses if is_taller else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer


def short_predicate(A, B):
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
    response_template = random.choice(true_responses if is_shorter else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer


def thin_predicate(A, B):
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
    response_template = random.choice(true_responses if is_thinner else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer


def small_predicate(A, B):
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
    response_template = random.choice(true_responses if is_smaller else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer


def behind_predicate(A, B):
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
    response_template = random.choice(true_responses if is_behind else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question + " Answer: " + answer


def front_predicate(A, B):
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

def left_choice(A, B):
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


def right_choice(A, B):
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


def above_choice(A, B):
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


def below_choice(A, B):
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


def tall_choice(A, B):
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


def short_choice(A, B):
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
    A, B, human_readable_dist, template_questions, template_answers
):
    A_desc, B_desc = A[0].lower(), B[0].lower()

    question_template = random.choice(template_questions)
    answer_template = random.choice(template_answers)

    # Replace placeholders with actual values
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = (
        answer_template.replace("[A]", A_desc)
        .replace("[B]", B_desc)
        .replace("[X]", human_readable_dist)
    )

    # Add to the dataset
    return question + " Answer: " + answer


def vertical_distance_data(A, B):
    template_questions = vertical_distance_questions
    template_answers = vertical_distance_answers

    A_center = A[1].get_axis_aligned_bounding_box().get_center()
    B_center = B[1].get_axis_aligned_bounding_box().get_center()
    vertical_distance = abs(A_center[1] - B_center[1])
    human_readable_dist = human_like_distance(vertical_distance)

    return generate_spatial_reasoning_data(
        A, B, human_readable_dist, template_questions, template_answers
    )


def horizontal_distance_data(A, B):
    template_questions = horizontal_distance_questions
    template_answers = horizontal_distance_answers

    A_center = A[1].get_axis_aligned_bounding_box().get_center()
    B_center = B[1].get_axis_aligned_bounding_box().get_center()
    horizontal_distance = np.sqrt(
        (A_center[0] - B_center[0]) ** 2 + (A_center[2] - B_center[2]) ** 2
    )

    human_readable_dist = human_like_distance(horizontal_distance)
    return generate_spatial_reasoning_data(
        A, B, human_readable_dist, template_questions, template_answers
    )


def width_data(A, B=None):
    A_desc = A[0].lower()

    template_questions = width_questions
    template_answers = width_answers

    width = A[1].get_axis_aligned_bounding_box().get_extent()[0]

    human_readable_width = human_like_distance(width)
    question_template = random.choice(template_questions)
    answer_template = random.choice(template_answers)

    question = question_template.replace("[A]", A_desc)
    answer = answer_template.replace("[A]", A_desc).replace("[X]", human_readable_width)

    return question + " Answer: " + answer

def height_data(A, B=None):
    A_desc = A[0].lower()

    template_questions = height_questions
    template_answers = height_answers

    width = A[1].get_axis_aligned_bounding_box().get_extent()[0]

    human_readable_width = human_like_distance(width)
    question_template = random.choice(template_questions)
    answer_template = random.choice(template_answers)

    question = question_template.replace("[A]", A_desc)
    answer = answer_template.replace("[A]", A_desc).replace("[X]", human_readable_width)

    return question + " Answer: " + answer


def evaluate_predicates_on_pairs(pairs, is_canonicalized):
    all_prompt_variants = [
        left_predicate,
        right_predicate,
        wide_predicate,
        big_predicate,
        thin_predicate,
        small_predicate,
        behind_predicate,
        front_predicate,
        left_choice,
        right_choice,
    ]

    add_canonicalized = [
            tall_choice,
            above_predicate,
            below_predicate,
            short_choice,
            below_choice,
            tall_predicate,
            short_predicate,
            above_choice,
            vertical_distance_data,
            horizontal_distance_data,
            width_data,
            height_data,
        ]

    if is_canonicalized:
        all_prompt_variants += add_canonicalized

    selected_predicates_choices = random.sample(all_prompt_variants, 10)

    results = []

    for A, B in pairs:
        pair_results = []
        for prompt_func in selected_predicates_choices:
            pair_results.append(prompt_func(A, B))

        # Run each of the distance functions
        distance = calculate_distances_between_point_clouds(A[1], B[1])
        pair_results.append(
            generate_spatial_reasoning_data(
                A,
                B,
                distance,
                distance_template_questions,
                distance_template_answers,
            )
        )

        results.extend(pair_results)
    return results
