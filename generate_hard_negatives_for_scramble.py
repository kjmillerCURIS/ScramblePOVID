import os
import sys
import copy
import json
import pickle
import random
import re
from tqdm import tqdm
from better_hard_negative_prompt import BETTER_HARD_NEGATIVE_PROMPT
from augment_data import augment_data_one, extract_modification_types


FAVORED_MODIFICATION_TYPES = ['Object Comparison Change', 'Verification Flip', 'Logical Operation Change']
FAVOR_WEIGHT = 0.8
MAX_NUM_ATTEMPTS = 10
DATA_STRIDE = 25
SAVE_FREQ = 50
BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/imgeneval-data')
SCRAMBLE_JSON_PATH = 'data/preference_data/coco_train_syn_cot_adv_ref_preference.json'
PARAMS = {'type_sensitive' : True, 'max_num_aug_attempts' : MAX_NUM_ATTEMPTS}


def create_augmentation_policy():
    modification_types = extract_modification_types(BETTER_HARD_NEGATIVE_PROMPT)
    assert(all([k in modification_types for k in FAVORED_MODIFICATION_TYPES]))
    beta = (1 - FAVOR_WEIGHT) / len(modification_types)
    alpha = beta + FAVOR_WEIGHT / len(FAVORED_MODIFICATION_TYPES)
    mixing_weights = {k : (alpha if k in FAVORED_MODIFICATION_TYPES else beta) for k in modification_types}
    return {'prompt' : BETTER_HARD_NEGATIVE_PROMPT, 'mixing_weights' : mixing_weights}


def get_output_filename(offset):
    return os.path.join(BASE_DIR, 'scramble_reweighted_hard_negatives/scramble_reweighted_hard_negatives_%d.json'%(offset))


def load_scramble_data(offset):
    with open(SCRAMBLE_JSON_PATH, 'r') as f:
        scramble_data = json.load(f)

    return scramble_data[offset::DATA_STRIDE]


def get_caption(scramble_datum):
    captions = [c['value'] for c in scramble_datum['conversations'] if c['from'] == 'gpt']
    assert(len(captions) == 1)
    return captions[0]


def generate_hard_negatives_for_scramble(offset):
    offset = int(offset)

    scramble_data = load_scramble_data(offset)
    output_filename = get_output_filename(offset)
    augmentation_policy = create_augmentation_policy()
    output = []
    start_index = 0
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            output = json.load(f)

        start_index = len(output)
        print('oh goody, already processed %d captions!'%(len(output)))

    for scramble_datum in tqdm(scramble_data[start_index:]):
        caption = get_caption(scramble_datum)
        negative_caption, modification_type, num_attempts = augment_data_one(PARAMS, caption, augmentation_policy)
        o = copy.deepcopy(scramble_datum)
        if negative_caption is None:
            print('??')
            o.pop('rejected_conversations')
            output.append(o)
            continue

        o['rejected_conversations'] = [{'from' : 'gpt', 'value' : negative_caption}]
        o['modification_type'] = modification_type
        o['num_attempts'] = num_attempts
        output.append(o)
        if len(output) % SAVE_FREQ == 0 and len(output) > 0:
            with open(output_filename, 'w') as f:
                json.dump(output, f)

    with open(output_filename, 'w') as f:
        json.dump(output, f)


def usage():
    print('Usage: python generate_hard_negatives_for_scramble.py <offset>')


if __name__ == '__main__':
    generate_hard_negatives_for_scramble(*(sys.argv[1:]))
