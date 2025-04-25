# 15 in-context examples below : 8 positive and 7 negative

# TODO : add negative examples
# TODO : try handpicked negative examples

def proc_str(s):
    s = s.capitalize()
    return s + '.' if not s.endswith('.') else s

WINOGROUND_IN_CONTEXT_PROMPT = """
Given an input sentence describing a scene, your task is to rearrange words in it to make a new sentence. The new sentence must meet the following three requirements:
1. The new sentence must be describing a different scene from the input sentence.
2. The new sentence must be fluent and grammatically correct.
3. The new sentence must make logical sense.

To complete the task, you should:
1. Answer the question of whether generating such a new sentence is possible using Yes or No.
2. Output the new sentence.

Here are some examples:
Input: A cat resting on a laptop next to a person.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: A cat resting on a person next to a laptop.

Input : A couple of large blue airplanes on a lot.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA 

Input: The cube is smaller than the shape whose lateral faces meet at a vertex.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: The shape whose lateral faces meet at a vertex is smaller than the cube.

Input: A bunch of bananas still attached to each other.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: The large person is drinking from the small coffee cup.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: The small person is drinking from the large coffee cup.

Input: An old green steam engine in the country.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: A person without glasses pushes a person with glasses sitting in a box.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: A person with glasses pushes a person without glasses sitting in a box.

Input: A bathroom with a toilet and a very dirty bathtub.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: The outer bristles are blue and the inner ones are white.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: The outer bristles are white and the inner ones are blue.

Input: A group of colorful umbrellas flying in the sky above a city.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: The circular mirror is on the left and the rectangular mirror is on the right.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: The rectangular mirror is on the left and the circular mirror is on the right.

Input: A mint green bus rides down the street.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: The triangular shape is beneath the square one.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: The square shape is beneath the triangular one.

Input: Friends sitting around a living room with their dog.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: The person is jumping while the cat is sitting.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: The person is sitting while the cat is jumping.

Input: {caption}
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense?
"""

# v2 has some handpicked winoground examples replacing random examples above 
WINOGROUND_IN_CONTEXT_PROMPT_V2 = """

Given an input sentence describing a scene, your task is to rearrange words in it to make a new sentence. The new sentence must meet the following three requirements:
1. The new sentence must be describing a different scene from the input sentence.
2. The new sentence must be fluent and grammatically correct.
3. The new sentence must make logical sense.

To complete the task, you should:
1. Answer the question of whether generating such a new sentence is possible using Yes or No.
2. Output the new sentence.

Here are some examples:
Input: A cat resting on a laptop next to a person.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: A cat resting on a person next to a laptop.

Input : A couple of large blue airplanes on a lot.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA 

Input: The outer bristles are blue and the inner ones are white.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: The outer bristles are white and the inner ones are blue.

Input: A bunch of bananas still attached to each other.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: The large person is drinking from the small coffee cup.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: The small person is drinking from the large coffee cup.

Input: An old green steam engine in the country.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: A bottle is in water.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: Water is in a bottle.

Input: A bathroom with a toilet and a very dirty bathtub.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: I had cleaned my car.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: I had my car cleaned.

Input: A group of colorful umbrellas flying in the sky above a city.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: The circular mirror is on the left and the rectangular mirror is on the right.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: The rectangular mirror is on the left and the circular mirror is on the right.

Input: A mint green bus rides down the street.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: There are three bananas and two apples.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: There are two bananas and three apples.

Input: Friends sitting around a living room with their dog.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? No
Output: NA

Input: The person is jumping while the cat is sitting.
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense? Yes
Output: The person is sitting while the cat is jumping.

Input: {caption}
Is it possible to rearrange the words in the input sentence to generate a new sentence that describes a different scene from the input sentence and makes logical sense?

"""

# addnl_pos_examples = """
# Input: concrete floors with wood walls
# Is it possible to rearrange the words in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
# Output: wood floors with concrete walls

# Input: the water is filled with plastic
# Is it possible to rearrange the words in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
# Output: the plastic is filled with water

# Input: a brown dog is on a white couch
# Is it possible to rearrange the words in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
# Output: a white dog is on a brown couch

# Input: the green one is fast and the one in white is comparatively slow
# Is it possible to rearrange the words in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
# Output: the green one is slow and the one in white is comparatively fast

# Input: a person wearing a bear mask in blue on the left hand side of a person wearing a panda mask with glasses in pink
# Is it possible to rearrange the words in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
# Output: a person wearing a panda mask with glasses in blue on the left hand side of a person wearing a bear mask in pink

# Input: one green apple surrounded by a bunch of red apples
# Is it possible to rearrange the words in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
# Output: one red apple surrounded by a bunch of green apples

# Input: the dog sits and the cat stands
# Is it possible to rearrange the words in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
# Output: the dog stands and the cat sits

# Input: the text is black on white
# Is it possible to rearrange the words in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? Yes
# Output: the text is white on black
# """

winoground_handpicked_egs = [
    {   
        "caption": "I had cleaned my car",
        "neg_caption": "I had my car cleaned"
    },  
    {   
        "caption": "a bottle is in water",
        "neg_caption": "water is in a bottle"
    },  
    {   
        "caption": "there are three bananas and two apples",
        "neg_caption": "there are two bananas and three apples"
    },  
    {   
        "caption": "the happy person is on the right and the sad person is on the left",
        "neg_caption": "the sad person is on the right and the happy person is on the left"
    },  
    {   
        "caption": "that person dusting off their hands",
        "neg_caption": "that person hands off their dusting"
    },  
    {   
        "caption": "the red car is behind the blue car",
        "neg_caption": "the blue car is behind the red car"
    },  
    {   
        "caption": "there are more skiers than snowboarders",
        "neg_caption": "there are more snowboarders than skiers"
    },  
    {   
        "caption": "a pink bird with a white beak",
        "neg_caption": "a white bird with a pink beak"
    },  
    {   
        "caption": "fishing for compliments",
        "neg_caption": "compliments for fishing"
    },  
    {   
        "caption": "some plants surrounding a lightbulb",
        "neg_caption": "a lightbulb surrounding some plants"
    }   
]