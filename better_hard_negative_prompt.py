import os
import sys


BETTER_HARD_NEGATIVE_PROMPT = '''
    Given the following input sentence describing a scene:
        "{CAPTION}"
    your task is to generate a hard negative — a fluent and grammatically correct sentence that describes a different scene from the input.
    You must modify the input according to the specified modification type:
        "{TYPE}"

    The new sentence you generate must satisfy all of the following:

    1. The new sentence must describe a different scene from the input sentence.
    2. The new sentence must be fluent and grammatically correct.
    3. The new sentence must make logical sense.

    Make sure you satisfy these requirements.
    Allowed Modification Types:
    **Subject Swap**:

    * Definition: Swap the subject noun phrase with another entity, keeping the verb and object the same. (Who is doing the action changes.)
    * Example: "A woman cuts a cake with a man behind her." → "A man cuts a cake with a woman behind him."

    **Object Swap**:

    * Definition: Swap the object noun phrase with another entity, keeping the verb and subject the same. (Who is receiving the action changes.)
    * Example: "A dog is chasing a cat." → "A cat is chasing a dog."

    **Attribute Swap**:

    * Definition: Exchange the attributes (e.g., color, size) of two different objects.
    * Example: "The red apple is beside the green pear." → "The green apple is beside the red pear."

    **Relation Swap**:

    * Definition: Replace the spatial or relational predicate with its logical opposite (e.g., behind → in front of).
    * Example: "The cat is behind the dog." → "The cat is in front of the dog."

    **Attribute Change**:

    * Definition: Change an object's attribute to a different plausible attribute (e.g., teal → white).
    * Example: "The bowl is teal." → "The bowl is white."

    **Relation Change**:

    * Definition: Change the spatial relation word to its opposite (e.g., under → on top of).
    * Example: "The books are under the chair." → "The books are on top of the chair."

    **Attribute + Relation Change**:

    * Definition: Simultaneously change both the attribute and the relation between objects.
    * Example: "The red apple is left of the green bowl." → "The green apple is right of the red bowl."

    **Object Count Change**:

    * Definition: Alter the number of mentioned objects (increase or decrease the count).
    * Example: "There are two jars on the shelf." → "There are three jars on the shelf."

    **Object Comparison Change**:

    * Definition: Change the comparative quantifier between two object groups (e.g., more → fewer).
    * Example: "There are more apples than jars." → "There are fewer apples than jars."

    **Verification Flip**:

    * Definition: Flip the existential quantifier (e.g., "no" ↔ "at least one").
    * Example: "There is no dog in the yard." → "There is at least one dog in the yard."

    **Logical Operation Change**:

    * Definition: Change logical operations from AND (both) to XOR (either) or vice versa.
    * Example: "There are both a chair and a desk in the room." → "There is either a chair or a desk in the room."

    Instructions:

    a.) Determine whether it is possible to generate a hard negative of the specified modification type.
    b.) Answer "Yes" or "No".
    c.) If Yes:
        i.) Specify the Modification Type.
        ii.) Specify the Elements to Modify.
        iii.) Output the New Sentence.
    d.) If No:
        i.) Output the sentence: "Not possible."

    Output Format:

    * Possible?: Yes / No
    * Modification Type: (one of the allowed types)
    * Elements to Modify: (what to change, if possible)
    * Output Sentence: (the new sentence, or "Not possible.")

    Do not add any explanations.
    In-Context Examples:
    Example 1 (Subject Swap) — Possible
    Input: A woman cutting into a cake with a man standing behind her.
    Modification Type: Subject Swap
    Possible?: Yes
    Elements to Modify: "a woman" ↔ "a man"
    Output Sentence: A man cutting into a cake with a woman standing behind him.
    Example 2 (Object Swap) — Possible
    Input: A dog sitting next to a bowl of water.
    Modification Type: Object Swap
    Possible?: Yes
    Elements to Modify: "dog" ↔ "bowl of water"
    Output Sentence: A bowl of water sitting next to a dog.
    Example 3 (Attribute Swap) — Possible
    Input: The red apple is on the table next to the green pear.
    Modification Type: Attribute Swap
    Possible?: Yes
    Elements to Modify: "red" ↔ "green"
    Output Sentence: The green apple is on the table next to the red pear.
    Example 4 (Object Count Change) — Not Possible
    Input: There is one sun in the sky.
    Modification Type: Object Count Change
    Possible?: No
    Output Sentence: Not possible.
    Example 5 (Logical Operation Change) — Not Possible
    Input: The apple is red.
    Modification Type: Logical Operation Change
    Possible?: No
    Output Sentence: Not possible.
    Example 6 (Logical Operation Change) — Possible
    Input: There are both a chair and a desk in the room.
    Modification Type: Logical Operation Change
    Possible?: Yes
    Elements to Modify: "both a chair and a desk" → "either a chair or a desk"
    Output Sentence: There is either a chair or a desk in the room.
'''

