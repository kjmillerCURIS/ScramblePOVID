# TODO : Decide how best to incorporate reasoning. Also should swaps that do not change the scene be caught at this stage or while filtering?

SWAP_OBJ_PROMPT_W_REASONING = """
Given an input sentence describing a scene, your task is to first locate two swappable noun phrases in the sentence, and then swap them to make a new sentence. The new sentence must meet the following three requirements:
1. The new sentence must be describing a different scene from the input sentence.
2. The new sentence must be fluent and grammatically correct.
3. The new sentence must make logical sense.

To complete the task, you should:
1. Answer the question of whether generating such a new sentence is possible using Yes or No.
2. Output the swappable noun phrases.
3. Output any additional reasoning if necessary.
4. Swap the selected noun phrases to generate a new sentence.

Here are some examples:

Input: A cat resting on a laptop next to a person.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable noun phrases: laptop, person
Additional reasoning: NA
Output: A cat resting on a person next to a laptop.

Input: A person standing by some very cute tall giraffes.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? No
Swappable noun phrases: some very cute tall giraffes, person
Additional reasoning: The new sentence "Some very cute tall giraffes standing by a person." describes the same scene as the input sentence.
Output: NA

Input: An old person kisses a young person.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable noun phrases: young person, old person
Additional reasoning: NA
Output: A young person kisses an old person.

Input: Surfers walking on the beach carrying surfboards.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? No
Swappable noun phrases: surfers, surfboards
Additional reasoning: The new sentence "Surfboards walking on a beach carrying surfers." does not make logical sense in a realistic scenario.
Output: NA

Input: The person without earrings pays the person with earrings.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable noun phrases: person without earrings, person with earrings
Additional reasoning: NA
Output: The person with earrings pays the person without earrings.

Input: A young zebra is sniffing the ground in a dusty area.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? No
Swappable noun phrases: young zebra, dusty area
Additional reasoning: The new sentence "A dusty area is sniffing the ground in a young zebra." does not make logical sense.
Output: NA

Input: The dog's leg is on the person's torso.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable noun phrases: dog, person
Additional reasoning: NA
Output: The person's leg is on the dog's torso.

Input: {caption}
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense?
"""


SWAP_OBJ_PROMPT = """
Given an input sentence describing a scene, your task is to first locate two swappable noun phrases in the sentence, and then swap them to make a new sentence. The new sentence must meet the following three requirements:
1. The new sentence must be describing a different scene from the input sentence.
2. The new sentence must be fluent and grammatically correct.
3. The new sentence must make logical sense.

To complete the task, you should:
1. Answer the question of whether generating such a new sentence is possible using Yes or No.
2. Output the swappable noun phrases.
3. Swap the selected noun phrases to generate a new sentence.

Here are some examples:

Input: A cat resting on a laptop next to a person.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable noun phrases: laptop, person
Output: A cat resting on a person next to a laptop.

Input: A person standing by some very cute tall giraffes.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? No
Swappable noun phrases: NA
Output: NA

Input: An old person kisses a young person.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable noun phrases: young person, old person
Output: A young person kisses an old person.

Input: A plate of food that contains smothered meat, potatoes, and broccoli.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? No
Swappable noun phrases: NA
Output: NA

Input: The person without earrings pays the person with earrings.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable noun phrases: person without earrings, person with earrings
Output: The person with earrings pays the person without earrings.

Input: A young zebra is sniffing the ground in a dusty area.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? No
Swappable noun phrases: NA
Output: NA

Input: The dog's leg is on the person's torso.
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable noun phrases: dog, person
Output: The person's leg is on the dog's torso.

Input: {caption}
Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense?
"""

SWAP_OBJ_PROMPT_NO_INCONTEXT = """
Given an input sentence describing a scene, your task is to first locate two swappable noun phrases in the sentence, and then swap them to make a new sentence. The new sentence must meet the following three requirements:
1. The new sentence must be describing a different scene from the input sentence.
2. The new sentence must be fluent and grammatically correct.
3. The new sentence must make logical sense.

To complete the task, you should:
1. Answer the question of whether generating such a new sentence is possible using Yes or No.
2. Output the swappable noun phrases.
3. Swap the selected noun phrases to generate a new sentence.

Input: {caption}
"""