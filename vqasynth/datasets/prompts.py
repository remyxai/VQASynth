import random
import numpy as np
from itertools import combinations


def tall_choice(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    result = "taller" if height_A > height_B else "taller" if height_B > height_A else "equally tall"
    return f"Which object is taller? Answer: {A_desc if height_A > height_B else B_desc if height_B > height_A else 'Neither'} is {result}."

def gap(A, B):
    # Placeholder for simplicity as gap calculation requires complex spatial analysis
    return "Gap estimation between two objects requires complex spatial analysis not implemented in this function."

def tall_short_classify(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    relation = "taller than" if height_A > height_B else "shorter than" if height_A < height_B else "equally tall as"
    return f"Comparing heights, {A_desc} is {relation} {B_desc}."

def above_predicate(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    min_z_A = A_cloud.get_axis_aligned_bounding_box().get_min_bound()[2]
    max_z_B = B_cloud.get_axis_aligned_bounding_box().get_max_bound()[2]
    return f"Is {A_desc} above {B_desc}? Answer: {'Yes' if min_z_A > max_z_B else 'No'}."

def below_predicate(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    max_z_A = A_cloud.get_axis_aligned_bounding_box().get_max_bound()[2]
    min_z_B = B_cloud.get_axis_aligned_bounding_box().get_min_bound()[2]
    return f"Is {A_desc} below {B_desc}? Answer: {'Yes' if max_z_A < min_z_B else 'No'}."

def short_choice(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    result = "shorter" if height_A < height_B else "shorter" if height_B < height_A else "equally short"
    return f"Which object is shorter? Answer: {A_desc if height_A < height_B else B_desc if height_B < height_A else 'Neither'} is {result}."

def below_diff(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    min_z_A = A_cloud.get_axis_aligned_bounding_box().get_min_bound()[2]
    min_z_B = B_cloud.get_axis_aligned_bounding_box().get_min_bound()[2]
    diff = abs(min_z_A - min_z_B)
    return f"The difference in elevation from the bottom of {A_desc} to the bottom of {B_desc} is {diff} units."

def tall_predicate(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    return f"Is {A_desc} taller than {B_desc}? Answer: {'Yes' if height_A > height_B else 'No'}."

def short_predicate(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    return f"Is {A_desc} shorter than {B_desc}? Answer: {'Yes' if height_A < height_B else 'No'}."

def above_diff(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    max_z_A = A_cloud.get_axis_aligned_bounding_box().get_max_bound()[2]
    max_z_B = B_cloud.get_axis_aligned_bounding_box().get_max_bound()[2]
    diff = abs(max_z_A - max_z_B)
    return f"The vertical distance from the top of {A_desc} to the top of {B_desc} is {diff} units."

def vertical_dist(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    center_z_A = A_cloud.get_center()[2]
    center_z_B = B_cloud.get_center()[2]
    distance = abs(center_z_A - center_z_B)
    return f"The vertical distance between {A_desc} and {B_desc} is {distance} units."

def horizontal_dist(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    center_A = A_cloud.get_center()
    center_B = B_cloud.get_center()
    distance = np.sqrt((center_A[0] - center_B[0])**2 + (center_A[1] - center_B[1])**2)
    return f"The horizontal distance between {A_desc} and {B_desc} is {distance} units."

def above_below_classify(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    if above_predicate(A, B).endswith("Yes"):
        relation = "above"
    elif below_predicate(A, B).endswith("Yes"):
        relation = "below"
    else:
        relation = "neither above nor below"
    return f"In terms of elevation, {A_desc} is {relation} {B_desc}."

def height(A):
    A_desc, A_cloud = A
    height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    return f"The height of {A_desc} is {height_A} units."

def width(A):
    A_desc, A_cloud = A
    width_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[0]
    return f"The width of {A_desc} is {width_A} units."

def above_choice(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    if above_predicate(A, B).endswith("Yes"):
        chosen = A_desc
    elif below_predicate(A, B).endswith("Yes"):
        chosen = B_desc
    else:
        chosen = "neither"
    return f"Which object is more above? Answer: {chosen}."

def below_choice(A, B):
    A_desc, A_cloud = A
    B_desc, B_cloud = B
    if below_predicate(A, B).endswith("Yes"):
        chosen = A_desc
    elif above_predicate(A, B).endswith("Yes"):
        chosen = B_desc
    else:
        chosen = "neither"
    return f"Which object is more below? Answer: {chosen}."

def elevation(A):
    A_desc, A_cloud = A
    min_z_A = A_cloud.get_axis_aligned_bounding_box().get_min_bound()[2]
    return f"The elevation of {A_desc} from the reference plane is {min_z_A} units."

def human_like_distance(distance_meters):
    # Define the choices with units included
    if distance_meters < 1:
        choices = [
            (0.5, "meters", 0.25),
            (round(distance_meters, 4), "meters", 0.75),
            (distance_meters * 3.28084, "feet", 0.15),
            (distance_meters * 39.3701, "inches", 0.1)
        ]
    elif distance_meters < 10:
        choices = [
            (round(distance_meters, 4), "meters", 0.8),
            (round(distance_meters * 3.28084, 4), "feet", 0.2)
        ]
    elif distance_meters < 100:
        choices = [
            (5 * round(distance_meters / 5, 4), "meters", 0.5),
            (10 * round(distance_meters / 10, 4), "meters", 0.3),
            (distance_meters * 1.09361, "yards", 0.2)
        ]
    else:
        choices = [
            (10 * round(distance_meters / 10, 4), "meters", 0.4),
            (50 * round(distance_meters / 50, 4), "meters", 0.3),
            (100 * round(distance_meters / 100, 4), "meters", 0.2),
            (distance_meters * 0.000621371, "miles", 0.1)
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
            #return f"{value} {unit}"
            return value, unit

    # Fallback to the last choice if something goes wrong
    #return f"{choices[-1][0]} {choices[-1][1]}"
    return choices[-1][0], choices[-1][1]

def evaluate_predicates_on_pairs(pairs):
    all_predicates = [tall_choice, gap, tall_short_classify, above_predicate, below_predicate,
                      short_choice, below_diff, tall_predicate, short_predicate, above_diff,
                      vertical_dist, horizontal_dist, above_below_classify, height, width,
                      above_choice, below_choice, elevation]
    
    selected_predicates = random.sample(all_predicates, 5)
    
    results = []
    
    for A, B in pairs:
        pair_results = []
        for predicate in selected_predicates:
            try:
                pair_results.append(predicate(A, B))
            except:
                pair_results.append(predicate(A))

        results.extend(pair_results)
    return results
