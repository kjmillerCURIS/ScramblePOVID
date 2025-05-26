import os
import sys
import json
from tqdm import tqdm
from run_grounding_dino_on_full_dataset import BASE_DIR, DATA_STRIDE, get_output_dict_filename
from run_groundingdino_simulation import make_decision


def apply_groundingdino_strategy(threshold, strategy):
    threshold = float(threshold)

    output_dict = {}
    for offset in range(DATA_STRIDE):
        with open(get_output_dict_filename(threshold, offset), 'r') as f:
            output_dict_one = json.load(f)

        for image_base in sorted(output_dict_one.keys()):
            output_dict[image_base] = output_dict_one[image_base]

    accepted = []
    rejected = []
    for image_base in tqdm(sorted(output_dict.keys())):
        is_accepted = make_decision(output_dict[image_base], strategy)
        if is_accepted:
            accepted.append(image_base)
        else:
            rejected.append(image_base)

    decision_filename = os.path.join(BASE_DIR, 'grounding_dino_decisions', 'grounding_dino_decisions_threshold%s_%s.json'%(str(threshold), strategy))
    os.makedirs(os.path.dirname(decision_filename), exist_ok=True)
    with open(decision_filename, 'w') as f:
        json.dump({'accepted' : accepted, 'rejected' : rejected}, f)


def usage():
    print('Usage: python apply_groundingdino_strategy.py <threshold> <strategy>')


if __name__ == '__main__':
    apply_groundingdino_strategy(*(sys.argv[1:]))
