import random
import numpy as np
from itertools import combinations
from vqasynth.prompt_templates import *
from vqasynth.scene_fusion import SpatialSceneConstructor 

class PromptGenerator():
    def __init__(self):
        self.spatial_scene_constructor = SpatialSceneConstructor()

    def human_like_distance(self, distance_meters):
        # Define the choices with units included, focusing on the 0.1 to 10 meters range
        if distance_meters < 1:  # For distances less than 1 meter
            choices = [
                (
                    round(distance_meters * 100, 2),
                    "centimeters",
                    0.2,
                ),  # Centimeters for very small distances
                (
                    round(distance_meters * 39.3701, 2),
                    "inches",
                    0.8,
                ),  # Inches for the majority of cases under 1 meter
            ]
        elif distance_meters < 3:  # For distances less than 3 meters
            choices = [
                (round(distance_meters, 2), "meters", 0.5),
                (
                    round(distance_meters * 3.28084, 2),
                    "feet",
                    0.5,
                ),  # Feet as a common unit within indoor spaces
            ]
        else:  # For distances from 3 up to 10 meters
            choices = [
                (
                    round(distance_meters, 2),
                    "meters",
                    0.7,
                ),  # Meters for clarity and international understanding
                (
                    round(distance_meters * 3.28084, 2),
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
                return f"{value} {unit}"

        # Fallback to the last choice if something goes wrong
        return f"{choices[-1][0]} {choices[-1][1]}"

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
        response_template = random.choice(true_responses if is_left else false_responses)

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
        response_template = random.choice(true_responses if is_right else false_responses)

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
        response_template = random.choice(true_responses if is_above else false_responses)

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
        response_template = random.choice(true_responses if is_below else false_responses)

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
        response_template = random.choice(true_responses if is_wider else false_responses)

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
        response_template = random.choice(true_responses if is_bigger else false_responses)

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
        response_template = random.choice(true_responses if is_taller else false_responses)

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
        response_template = random.choice(true_responses if is_shorter else false_responses)

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
        response_template = random.choice(true_responses if is_thinner else false_responses)

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
        response_template = random.choice(true_responses if is_smaller else false_responses)

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
        response_template = random.choice(true_responses if is_behind else false_responses)

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

        # Replace placeholders with actual values
        question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = (
            answer_template.replace("[A]", A_desc)
            .replace("[B]", B_desc)
            .replace("[X]", human_readable_dist)
        )

        # Add to the dataset
        return question + " Answer: " + answer


    def vertical_distance_data(self, A, B):
        template_questions = vertical_distance_questions
        template_answers = vertical_distance_answers

        A_center = A[1].get_axis_aligned_bounding_box().get_center()
        B_center = B[1].get_axis_aligned_bounding_box().get_center()
        vertical_distance = abs(A_center[1] - B_center[1])
        human_readable_dist = self.human_like_distance(vertical_distance)

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

        human_readable_dist = self.human_like_distance(horizontal_distance)
        return self.generate_spatial_reasoning_data(
            A, B, human_readable_dist, template_questions, template_answers
        )


    def width_data(self, A, B=None):
        A_desc = A[0].lower()

        template_questions = width_questions
        template_answers = width_answers

        width = A[1].get_axis_aligned_bounding_box().get_extent()[0]

        human_readable_width = self.human_like_distance(width)
        question_template = random.choice(template_questions)
        answer_template = random.choice(template_answers)

        question = question_template.replace("[A]", A_desc)
        answer = answer_template.replace("[A]", A_desc).replace("[X]", human_readable_width)

        return question + " Answer: " + answer

    def height_data(self, A, B=None):
        A_desc = A[0].lower()

        template_questions = height_questions
        template_answers = height_answers

        width = A[1].get_axis_aligned_bounding_box().get_extent()[0]

        human_readable_width = self.human_like_distance(width)
        question_template = random.choice(template_questions)
        answer_template = random.choice(template_answers)

        question = question_template.replace("[A]", A_desc)
        answer = answer_template.replace("[A]", A_desc).replace("[X]", human_readable_width)

        return question + " Answer: " + answer


    def evaluate_predicates_on_pairs(self, pairs, is_canonicalized):
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

        selected_predicates_choices = random.sample(all_prompt_variants, 10)

        results = []

        for A, B in pairs:
            pair_results = []
            for prompt_func in selected_predicates_choices:
                pair_results.append(prompt_func(A, B))

            # Run each of the distance functions
            distance = self.spatial_scene_constructor.calculate_distances_between_point_clouds(A[1], B[1])
            distance = self.human_like_distance(distance)
            pair_results.append(
                self.generate_spatial_reasoning_data(
                    A,
                    B,
                    distance,
                    distance_template_questions,
                    distance_template_answers,
                )
            )

            results.extend(pair_results)
        return results

    def run(self, captions, pointclouds, is_canonicalized):
        pointclouds = self.spatial_scene_constructor.restore_pointclouds(pointclouds)
        try:
            objects = list(zip(captions, pointclouds))
            all_pairs = [(i, j) for i in range(len(objects)) for j in range(len(objects)) if i != j]
            random.shuffle(all_pairs)
            selected_pairs = all_pairs[:5]
            object_pairs = [(objects[i], objects[j]) for i,j in selected_pairs]
            prompts = self.evaluate_predicates_on_pairs(object_pairs, is_canonicalized)
        except:
            prompts = []
        return prompts

    def apply_transform(self, example):
        """
        Process a single row in the dataset, adding depth map and focal length.

        Args:
            example: A single example from the dataset.
            images: The column in the dataset containing the images.

        Returns:
            Updated example with depth map and focal length.
        """
        try:
            example['prompts'] = self.run(
                example["captions"],
                example["pointclouds"],
                example["is_canonicalized"]
            )

            messages = []
            first_prompt = True

            for prompt in example['prompts']:
                if 'Answer: ' in prompt:
                    question, answer = prompt.split('Answer: ', 1)

                    # For the first prompt, include the image tag
                    if first_prompt:
                        messages.append({
                            "content": [{"index": 0, "text": None, "type": "image"}, {"index": None, "text": question.strip(), "type": "text"}],
                            "role": "user"
                        })
                    else:
                        messages.append({
                            "content": [{"index": None, "text": question.strip(), "type": "text"}],
                            "role": "user"
                        })

                    # Add assistant response
                    messages.append({
                        "content": [{"index": None, "text": answer.strip(), "type": "text"}],
                        "role": "assistant"
                    })
                    first_prompt = False

            example['messages'] = messages
        except Exception as e:
            print(f"Error processing image, skipping: {e}")
            example['prompts'] = None
            example['messages'] = None
        return example
