import os
import sys
import random
from openai import OpenAI
from huggingface_utils import HUGGINGFACE_API_KEY #CAUTION: please do NOT commit your copy of huggingface_utils, github won't let you because it has a key
from better_hard_negative_prompt import BETTER_HARD_NEGATIVE_PROMPT


client = OpenAI( api_key=HUGGINGFACE_API_KEY,
                 base_url="https://api.deepinfra.com/v1/openai",)


def ask_deepseek(prompt):
    response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                stream=False,
            )
    return response.choices[0].message.content


def extract_modification_types(prompt):
    lines = prompt.split('\n')
    modification_types = []
    for line in lines:
        if line.count('**') == 2:
            ss = line.split('**')
            assert(len(ss) == 3)
            if ss[1] not in modification_types:
                modification_types.append(ss[1])

    return modification_types


def initialize_augmentation_policy(params):
    augmentation_policy = {'prompt' : BETTER_HARD_NEGATIVE_PROMPT}
    if params['type_sensitive']:
        modification_types = extract_modification_types(augmentation_policy['prompt'])
        augmentation_policy['mixing_weights'] = {k : 1 / len(modification_types) for k in modification_types}

    return augmentation_policy


def extract_possible(reply):
    if 'output sentence:' not in reply.lower():
        return False

    lines = reply.lower().split('\n')
    for line in lines:
        if 'possible?:' in line:
            if ' yes' in line:
                return True
            else:
                return False

    return False


def augment_data_one(params, caption, augmentation_policy):
    prompt = augmentation_policy['prompt']
    if params['type_sensitive']:
        mixing_weights = augmentation_policy['mixing_weights']
    else:
        all_modification_types = extract_modification_types(prompt)
        print(all_modification_types)
        mixing_weights = {modification_type : 1 / len(modification_types) for modification_type in modification_types}

    modification_type = random.choices(sorted(mixing_weights.keys()), weights=[mixing_weights[k] for k in sorted(mixing_weights.keys())], k=1)[0]
    print(modification_type)
    query = prompt.format(ANCHOR=caption, TYPE=modification_type)
    num_attempts = 0
    while True:
        num_attempts += 1
        reply = ask_deepseek(query)
        is_possible = extract_possible(reply)
        print(is_possible)
        if is_possible:
            n = len(reply.lower().split('output sentence:')[-1])
            negative_example = reply[-n:].strip()
            print(negative_example)
            return negative_example, num_attempts
        else:
            print(reply)
            if num_attempts >= params['max_num_aug_attempts']:
                return None, num_attempts


#Step 4: use augmentation policy to augment data, add/replace to existing data_buffer, which will be returned
def augment_data(params, data_source, data_buffer, augmentation_policy):
    new_data = []
    for datum in data_source:
        new_datum = copy.deepcopy(datum)
        caption = datum['positive_caption'] #FIXME: double-check that this is actually what Samarth does, as opposed to augmenting an "anchor" to both T+ and T-
        negative_caption, _ = augment_data_one(params, caption, augmentation_policy)
        if negative_caption is None:
            print('??')
            continue

        new_datum['negative_caption'] = negative_caption
        new_data.append(new_datum)

    #now combine new data with old!
    N_new = int(round(params['keep_ratio_for_new_data'] * len(new_data)))
    N_old = int(round(params['keep_ratio_for_old_data'] * len(new_data)))
    updated_data_buffer = random.sample(new_data, N_new) + random.sample(data_buffer, N_old)
    return updated_data_buffer

    assert(False)
