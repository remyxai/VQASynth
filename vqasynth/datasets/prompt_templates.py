# Distance
distance_template_questions = [
    "What is the distance between the [A] and the [B]?",
    "How far apart are the [A] and the [B]?",
    "How distant is the [A] from the [B]?",
    "How far is the [A] from the [B]?",
    "How close is the [A] from the [B]?",
    "Could you measure the distance between the [A] and the [B]?",
    "Can you tell me the distance of the [A] from the [B]?",
    "How far away is the [A] from the [B]?",
    "Can you provide the distance measurement between the [A] and the [B]?",
    "Can you give me an estimation of the distance between the [A] and the [B]?",
    "Could you provide the distance between the [A] and the [B]?",
    "How much distance is there between the [A] and the [B]?",
    "Tell me the distance between the [A] and the [B].",
    "Give me the distance from the [A] to the [B].",
    "Measure the distance from the [A] to the [B].",
    "Measure the distance between the [A] and the [B].",
]

distance_template_answers = [
    "[X]",
    "the [A] and the [B] are [X] apart.",
    "the [A] is [X] away from the [B].",
    "A distance of [X] exists between the [A] and the [B].",
    "the [A] is [X] from the [B].",
    "the [A] and the [B] are [X] apart from each other.",
    "They are [X] apart.",
    "The distance of the [A] from the [B] is [X].",
]

# Predicates
left_predicate_questions = [
    "Is the [A] to the left of the [B] from the viewer's perspective?",
    "Does the [A] appear on the left side of the [B]?",
    "Can you confirm if the [A] is positioned to the left of the [B]?",
]

left_true_responses = [
    "Yes, the [A] is to the left of the [B].",
    "Indeed, the [A] is positioned on the left side of the [B].",
    "Correct, you'll find the [A] to the left of the [B].",
]

left_false_responses = [
    "No, the [A] is not to the left of the [B].",
    "In fact, the [A] is either to the right of or directly aligned with the [B].",
    "Incorrect, the [A] is not on the left side of the [B].",
]

right_predicate_questions = [
    "Is the [A] to the right of the [B] from the viewer's perspective?",
    "Does the [A] appear on the right side of the [B]?",
    "Can you confirm if the [A] is positioned to the right of the [B]?",
]

right_true_responses = [
    "Yes, the [A] is to the right of the [B].",
    "Indeed, the [A] is positioned on the right side of the [B].",
    "Correct, you'll find the [A] to the right of the [B].",
]

right_false_responses = [
    "No, the [A] is not to the right of the [B].",
    "In fact, the [A] is either to the left of or directly aligned with the [B].",
    "Incorrect, the [A] is not on the right side of the [B].",
]

above_predicate_questions = [
    "Is the [A] above the [B]?",
    "Does the [A] appear over the [B]?",
    "Can you confirm if the [A] is positioned above the [B]?",
]

above_true_responses = [
    "Yes, the [A] is above the [B].",
    "Indeed, the [A] is positioned over the [B].",
    "Correct, the [A] is located above the [B].",
]

above_false_responses = [
    "No, the [A] is not above the [B].",
    "Actually, the [A] is either below or at the same elevation as the [B].",
    "Incorrect, the [A] is not positioned above the [B].",
]

below_predicate_questions = [
    "Is the [A] below the [B]?",
    "Does the [A] appear under the [B]?",
    "Can you confirm if the [A] is positioned below the [B]?",
]

below_true_responses = [
    "Yes, the [A] is below the [B].",
    "Indeed, the [A] is positioned under the [B].",
    "Correct, the [A] is located below the [B].",
]

below_false_responses = [
    "No, the [A] is not below the [B].",
    "Actually, the [A] is either above or at the same elevation as the [B].",
    "Incorrect, the [A] is not positioned below the [B].",
]

wide_predicate_questions = [
    "Is the [A] wider than the [B]?",
    "Does the [A] have a greater width compared to the [B]?",
    "Can you confirm if the [A] is wider than the [B]?",
]

wide_true_responses = [
    "Yes, the [A] is wider than the [B].",
    "Indeed, the [A] has a greater width compared to the [B].",
    "Correct, the width of the [A] exceeds that of the [B].",
]

wide_false_responses = [
    "No, the [A] is not wider than the [B].",
    "In fact, the [A] might be narrower or the same width as the [B].",
    "Incorrect, the [A]'s width does not surpass the [B]'s.",
]

big_predicate_questions = [
    "Is the [A] bigger than the [B]?",
    "Does the [A] have a larger size compared to the [B]?",
    "Can you confirm if the [A] is bigger than the [B]?",
]

big_true_responses = [
    "Yes, the [A] is bigger than the [B].",
    "Indeed, the [A] has a larger size compared to the [B].",
    "Correct, the [A] is larger in size than the [B].",
]

big_false_responses = [
    "No, the [A] is not bigger than the [B].",
    "Actually, the [A] might be smaller or the same size as the [B].",
    "Incorrect, the [A] is not larger than the [B].",
]

tall_predicate_questions = [
    "Is the [A] taller than the [B]?",
    "Does the [A] have a greater height compared to the [B]?",
    "Can you confirm if the [A] is taller than the [B]?",
]

tall_true_responses = [
    "Yes, the [A] is taller than the [B].",
    "Indeed, the [A] has a greater height compared to the [B].",
    "Correct, the [A] is much taller as the [B].",
]

tall_false_responses = [
    "No, the [A] is not taller than the [B].",
    "In fact, the [A] may be shorter or the same height as the [B].",
    "Incorrect, the [A]'s height is not larger of the [B]'s.",
]

short_predicate_questions = [
    "Is the [A] shorter than the [B]?",
    "Does the [A] have a lesser height compared to the [B]?",
    "Can you confirm if the [A] is shorter than the [B]?",
]

short_true_responses = [
    "Yes, the [A] is shorter than the [B].",
    "Indeed, the [A] has a lesser height compared to the [B].",
    "Correct, the [A] is not as tall as the [B].",
]

short_false_responses = [
    "No, the [A] is not shorter than the [B].",
    "In fact, the [A] may be taller or the same height as the [B].",
    "Incorrect, the [A]'s height does not fall short of the [B]'s.",
]

thin_predicate_questions = [
    "Is the [A] thinner than the [B]?",
    "Does the [A] have a lesser width compared to the [B]?",
    "Can you confirm if the [A] is thinner than the [B]?",
]

thin_true_responses = [
    "Yes, the [A] is thinner than the [B].",
    "Indeed, the [A] has a lesser width compared to the [B].",
    "Correct, the [A]'s width is less than the [B]'s.",
]

thin_false_responses = [
    "No, the [A] is not thinner than the [B].",
    "In fact, the [A] might be wider or the same width as the [B].",
    "Incorrect, the [A]'s width is not less than the [B]'s.",
]

small_predicate_questions = [
    "Is the [A] smaller than the [B]?",
    "Does the [A] have a smaller size compared to the [B]?",
    "Can you confirm if the [A] is smaller than the [B]?",
]

small_true_responses = [
    "Yes, the [A] is smaller than the [B].",
    "Indeed, the [A] has a smaller size compared to the [B].",
    "Correct, the [A] occupies less space than the [B].",
]

small_false_responses = [
    "No, the [A] is not smaller than the [B].",
    "Actually, the [A] might be larger or the same size as the [B].",
    "Incorrect, the [A] is not smaller in size than the [B].",
]

behind_predicate_questions = [
    "Is the [A] behind the [B]?",
    "Is the position of the [A] more distant than that of the [B]?",
    "Does the [A] lie behind the [B]?",
    "Is the [A] positioned behind the [B]?",
    "Is the [A] further to camera compared to the [B]?",
    "Does the [A] come behind the [B]?",
    "Is the [A] positioned at the back of the [B]?",
    "Is the [A] further to the viewer compared to the [B]?",
]

behind_true = [
    "Yes.",
    "Yes, it is.",
    "Yes, it is behind the [B].",
    "That is True.",
    "Yes, the [A] is further from the viewer.",
    "Yes, the [A] is behind the [B].",
]

behind_false = [
    "No.",
    "No, it is not.",
    "No, it is in front of the [B].",
    "That is False.",
    "No, the [A] is closer to the viewer.",
    "No, the [B] is in front of the [A].",
]

front_predicate_questions = [
    "Is the [A] in front of the [B]?",
    "Is the position of the [A] less distant than that of the [B]?",
    "Does the [A] lie in front of the [B]?",
    "Is the [A] positioned in front of the [B]?",
    "Is the [A] closer to camera compared to the [B]?",
    "Does the [A] come in front of the [B]?",
    "Is the [A] positioned before the [B]?",
    "Is the [A] closer to the viewer compared to the [B]?",
]

front_true = [
    "Yes.",
    "Yes, it is.",
    "Yes, it is in front of the [B].",
    "That is True.",
    "Yes, the [A] is closer to the viewer.",
    "Yes, the [A] is in front of the [B].",
]

front_false = [
    "No.",
    "No, it is not.",
    "No, it is behind the [B].",
    "That is False.",
    "No, the [A] is further to the viewer.",
    "No, the [B] is behind the [A].",
]


# Choice
left_choice_questions = [
    "Which is more to the left, the [A] or the [B]?",
    "Between the [A] and the [B], which one appears on the left side from the viewer's perspective?",
    "Who is positioned more to the left, the [A] or the [B]?",
]

left_choice_responses = [
    "[X] is more to the left.",
    "From the viewer's perspective, [X] appears more on the left side.",
    "Positioned to the left is [X].",
]

right_choice_questions = [
    "Which is more to the right, the [A] or the [B]?",
    "Between the [A] and the [B], which one appears on the right side from the viewer's perspective?",
    "Who is positioned more to the right, the [A] or the [B]?",
]

right_choice_responses = [
    "[X] is more to the right.",
    "From the viewer's perspective, [X] appears more on the right side.",
    "Positioned to the right is [X].",
]

above_choice_questions = [
    "Which is above, the [A] or the [B]?",
    "Between the [A] and the [B], which one is positioned higher?",
    "Who is higher up, the [A] or the [B]?",
]

above_choice_responses = [
    "[X] is above.",
    "Positioned higher is [X].",
    "[X] is higher up.",
]

below_choice_questions = [
    "Which is below, the [A] or the [B]?",
    "Between the [A] and the [B], which one is positioned lower?",
    "Who is lower down, the [A] or the [B]?",
]

below_choice_responses = [
    "[X] is below.",
    "Positioned lower is [X].",
    "[X] is lower down.",
]

tall_choice_questions = [
    "Who is taller, the [A] or the [B]?",
    "Between the [A] and the [B], which one has more height?",
    "Which of these two, the [A] or the [B], stands taller?",
]

tall_choice_responses = [
    "[X] is taller.",
    "With more height is [X].",
    "Standing taller between the two is [X].",
]

short_choice_questions = [
    "Who is shorter, the [A] or the [B]?",
    "Between the [A] and the [B], which one has less height?",
    "Which of these two, the [A] or the [B], stands shorter?",
]

short_choice_responses = [
    "[X] is shorter.",
    "With less height is [X].",
    "Standing shorter between the two is [X].",
]


# Vertical and horizonal distance
vertical_distance_questions = [
    "What is the vertical distance between the [A] and the [B]?",
    "How far apart are the [A] and the [B] vertically?",
    "How distant is the [A] from the [B] vertically?",
    "How far is the [A] from the [B] vertically?",
    "Could you measure the vertical distance between the [A] and the [B]?",
    "Can you tell me the vertical distance between the [A] and the [B]?",
    "How far away is the [A] from the [B] vertically?",
    "Estimate the vertical distance between the [A] and the [B].",
    "Could you provide the vertical distance between the [A] and the [B]?",
    "How much distance is there between the [A] and the [B] vertically?",
    "Tell me the distance between the [A] and the [B] vertically.",
    "Give me the vertical distance from the [A] to the [B].",
    "Measure the vertical distance from the [A] to the [B].",
    "Measure the distance between the [A] and the [B] vertically.",
]

vertical_distance_answers = [
    "[X]",
    "the [A] and the [B] are [X] apart vertically.",
    "the [A] is [X] away from the [B] vertically.",
    "A vertical distance of [X] exists between the [A] and the [B].",
    "the [A] is [X] from the [B] vertically.",
    "the [A] and the [B] are [X] apart vertically from each other.",
    "Vertically, They are [X] apart.",
    "The vertical distance of the [A] from the [B] is [X].",
    "They are [X] apart.",
    "It is approximately [X].",
]

horizontal_distance_questions = [
    "What is the horizontal distance between the [A] and the [B]?",
    "How far apart are the [A] and the [B] horizontally?",
    "How distant is the [A] from the [B] horizontally?",
    "How far is the [A] from the [B] horizontally?",
    "Could you measure the horizontal distance between the [A] and the [B]?",
    "Can you tell me the horizontal distance of the [A] from the [B]?",
    "Can you give me an estimation of the horizontal distance between the [A]"
    " and the [B]?"
    "Could you provide the horizontal distance between the [A] and the [B]?",
    "How much distance is there between the [A] and the [B] horizontally?",
    "Tell me the distance between the [A] and the [B] horizontally.",
    "Give me the horizontal distance from the [A] to the [B].",
    "Vertial gap between the [A] and the [B].",
    "Measure the horizontal distance from the [A] to the [B].",
    "Measure the distance between the [A] and the [B] horizontally.",
]

horizontal_distance_answers = [
    "[X]",
    "the [A] and the [B] are [X] apart horizontally.",
    "the [A] is [X] away from the [B] horizontally.",
    "A horizontal distance of [X] exists between the [A] and the [B].",
    "the [A] is [X] from the [B] horizontally.",
    "the [A] and the [B] are [X] apart horizontally from each other.",
    "Horizontally, They are [X] apart.",
    "The horizontal distance of the [A] from the [B] is [X].",
    "They are [X] apart.",
    "It is approximately [X].",
]

# Width/Height
width_questions = [
    "Measure the width of the [A].",
    "Determine the horizontal dimensions of the [A].",
    "Find out how wide the [A] is.",
    "What is the width of the [A]?",
    "How wide is the [A]?",
    "What are the dimensions of the [A] in terms of width?",
    "Could you tell me the horizontal size of the [A]?",
    "What is the approximate width of the [A]?",
    "How wide is the [A]?",
    "How much space does the [A] occupy horizontally?",
    "How big is the [A]?",
    "How big is the [A] in terms of width?",
    "What is the radius of the [A]?",
]
width_answers = [
    "[X]",
    "The width of the [A] is [X].",
    "the [A] is [X] wide.",
    "the [A] is [X] in width.",
    "It is [X].",
]

height_questions = [
    "Measure the height of the [A].",
    "Determine the vertical dimensions of the [A].",
    "Find out how tall the [A] is.",
    "What is the height of the [A]?",
    "How tall is the [A]?",
    "What are the dimensions of the [A] in terms of height?",
    "Could you tell me the vericall size of the [A]?",
    "What is the approximate height of the [A]?",
    "How tall is the [A]?",
    "How much space does the [A] occupy vertically?",
    "How tall is the [A]?",
    "How tall is the [A] in terms of width?",
]
height_answers = [
    "[X]",
    "The height of the [A] is [X].",
    "the [A] is [X] tall.",
    "the [A] is [X] in height.",
    "It is [X].",
]
