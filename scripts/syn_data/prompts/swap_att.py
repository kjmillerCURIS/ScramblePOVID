# TODO : also add examples where an attribute could be associated with another noun. Maybe that is a different rule?

# Input: A large black dog is lying on a white comforter.
# Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
# Swappable attributes: black, white
# Output: A large white dog is lying on a black comforter.

SWAP_ATT_PROMPT = """
Given an input sentence describing a scene, your task is to first locate two swappable adjectives in the sentence describing different objects, and then swap them to make a new sentence.
The new sentence must meet the following three requirements:
1. The new sentence must be describing a different scene from the input sentence.
2. The new sentence must be fluent and grammatically correct.
3. The new sentence must make logical sense.

To complete the task, you should:
1. Answer the question of whether generating such a new sentence is possible using Yes or No.
2. Output the swappable adjectives.
3. Swap them to make a new sentence.

Here are some examples:

Input: A girl in a pink shirt holding a blue umbrella.
Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable attributes: pink, blue
Output: A girl in a blue shirt holding a pink umbrella.

Input: A car and a truck are going through the intersection.
Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? No
Swappable attributes: NA
Output: NA

Input: A cold drink on a hot day.
Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable attributes: cold, hot
Output: A hot drink on a cold day.

Input: Four yellow airplanes flying side by side at an air show.
Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? No
Swappable attributes: NA
Output: NA

Input: The dress on the left is long and the dress on the right is short.
Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable attributes: long, short
Output: The dress on the left is short and the dress on the right is long.

Input: A big grey elephant standing in the jungle.
Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? No
Swappable attributes: NA
Output: NA

Input: A large black dog is lying on a white comforter.
Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
Swappable attributes: black, white
Output: A large white dog is lying on a black comforter.

Input: {caption}
Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense?
"""

SWAP_ATT_PROMPT_NO_INCONTEXT = """
Given an input sentence describing a scene, your task is to first locate two swappable adjectives in the sentence describing different objects, and then swap them to make a new sentence.
The new sentence must meet the following three requirements:
1. The new sentence must be describing a different scene from the input sentence.
2. The new sentence must be fluent and grammatically correct.
3. The new sentence must make logical sense.

To complete the task, you should:
1. Answer the question of whether generating such a new sentence is possible using Yes or No.
2. Output the swappable adjectives.
3. Swap them to make a new sentence.

Input: {caption}
"""