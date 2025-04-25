COT_PROMPT = """
Given an input caption describing a scene, your task is to rearrange words in it to make a new caption. The new caption must meet the following three requirements:
1. It must describe a scene with visual differences compared to the scene described by the input caption.
2. It must be fluent and grammatically correct.
3. It must make logical sense.
Note that you can choose to abstain and output 'NA' if it is not possible to generate a negative caption for the given input.

Example 1:
Input Caption: a pink bird with a white beak
Reasoning: 

1. Identify the key elements: 
   - Color of the bird: pink
   - Color of the beak: white
   - Structure: "[color] bird with a [color] beak"

2. Recognize that the negative caption should describe a different visual image using the same words

3. Observe that the colors are the main distinguishing features

4. Swap the colors while maintaining the structure:
   - "pink" describing the bird becomes "white"
   - "white" describing the beak becomes "pink"

5. Keep the rest of the structure intact:
   - "a [color] bird with a [color] beak"

6. Apply the swapped colors to the structure:
   - "a white bird with a pink beak"
This transformation maintains the same words and grammatical structure but inverts the color assignments, creating a visually distinct image that serves as an effective negative caption.
Final Output Caption : a white bird with a pink beak

Example 2:
Input Caption: a bottle is in water
Reasoning:

1. Identify the key elements:
   - Object 1: bottle
   - Object 2: water
   - Relationship: "is in"

2. Recognize that the negative caption should describe a different visual image using the same words

3. Observe that the relationship between the bottle and water is the distinguishing feature

4. Swap the positions of "bottle" and "water" in the sentence:
   - "Bottle" moves from the subject to the object position
   - "Water" moves from the object to the subject position

5. Adjust the articles as needed:
   - "a bottle" remains "a bottle" (no change needed)
   - "water" remains "water" (no article needed for this uncountable noun)

6. Keep the relationship "is in" intact

7. Reconstruct the sentence with the swapped elements:
   - "water is in a bottle"

This transformation maintains the same core words but inverts the spatial relationship between the bottle and water. It creates a contrasting scenario that effectively serves as a negative caption, describing a completely different visual image while using the same words. 
Final Output Caption: water is in a bottle

Example 3:
Input Caption: Manning a ship.
Reasoning:

1. Identify the key words: "manning" and "ship"

2. Recognize that "manning" is a verb form derived from "man"

3. Swap the roles of "man" and "ship":
   - "Man" becomes the object
   - "Ship" becomes the basis for the verb

4. Transform "ship" into a verb:
   - "Ship" becomes "shipping"

5. Change "man" to its object form:
   - "Man" becomes "a man"

6. Rearrange the words to form a grammatically correct sentence:
   - "Shipping a man"

This transformation maintains the same core words but completely changes the meaning and visual imagery, creating a contrasting scenario that serves as an effective negative caption.
Final Output Caption: Shipping a man.

Example 4:
Input Caption: there are three bananas and two apples
Reasoning:

1. Identify the key elements:
   - Object 1: bananas
   - Object 2: apples
   - Quantity of bananas: three
   - Quantity of apples: two
   - Structure: "there are [number] [fruit] and [number] [fruit]"

2. Recognize that the negative caption should describe a different visual image using the same words

3. Observe that the quantities are the main distinguishing features

4. Swap the quantities while maintaining the fruits they describe:
   - "three" describing bananas becomes "two"
   - "two" describing apples becomes "three"

5. Keep the rest of the structure intact:
   - "there are" at the beginning
   - "and" connecting the two fruit phrases

6. Apply the swapped quantities to the structure:
   - "there are two bananas and three apples"

This transformation maintains the same words and grammatical structure but inverts the quantities assigned to each fruit. It creates a visually distinct image that serves as an effective negative caption, describing a different arrangement of fruits while using the exact same words as the positive caption.
Final Output Caption: there are two bananas and three apples

Example 5:
Input Caption: There are more skiers than snowboarders. 
Reasoning:

1. Identify the key elements:
   - Group 1: skiers
   - Group 2: snowboarders
   - Comparative structure: "more [group] than [group]"
   - Opening phrase: "There are"

2. Recognize that the negative caption should describe a different visual image using the same words

3. Observe that the comparative relationship between skiers and snowboarders is the distinguishing feature

4. Swap the positions of "skiers" and "snowboarders" in the sentence:
   - "skiers" moves from the subject of comparison to the object
   - "snowboarders" moves from the object of comparison to the subject

5. Keep the comparative structure "more ... than" intact

6. Maintain the opening phrase "There are"

7. Reconstruct the sentence with the swapped elements:
   - "There are more snowboarders than skiers"

This transformation maintains the same words and grammatical structure but inverts the comparative relationship between skiers and snowboarders. It creates a contrasting scenario that effectively serves as a negative caption, describing a completely different visual image (with snowboarders outnumbering skiers instead of vice versa) while using the exact same words as the positive caption.
Final Output Caption: There are more snowboarders than skiers.

Example 6:
Input Caption: {caption}
Reasoning:
"""