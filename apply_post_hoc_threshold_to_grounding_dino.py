import os
import sys
import json
from tqdm import tqdm
print('importing...')
from run_grounding_dino_on_full_dataset import DATA_STRIDE, get_output_dict_filename
print('done importing')


def apply_mask(old_output, mask, k):
    assert(len(old_output['detection_results'][k]) == len(mask))
    return [x for x, keep in zip(old_output['detection_results'][k], mask) if keep]


def modify_output(old_output, new_threshold):
    new_output = {}
    new_output['caption'] = old_output['caption']
    new_output['parser_results'] = old_output['parser_results']
    new_output['detection_results'] = {}
    mask = [score > new_threshold for score in old_output['detection_results']['scores']]
    new_output['detection_results']['scores'] = apply_mask(old_output, mask, 'scores')
    new_output['detection_results']['boxes'] = apply_mask(old_output, mask, 'boxes')
    new_output['detection_results']['text_labels'] = apply_mask(old_output, mask, 'text_labels')
    new_output['detection_results']['labels'] = apply_mask(old_output, mask, 'labels')
    return new_output


def apply_post_hoc_threshold_to_grounding_dino_one_offset(old_threshold, new_threshold, offset):
    old_output_dict_filename = get_output_dict_filename(old_threshold, offset)
    print('loading...')
    with open(old_output_dict_filename, 'r') as f:
        old_output_dict = json.load(f)

    print('done loading')
    new_output_dict_filename = get_output_dict_filename(new_threshold, offset)
    new_output_dict = {}
    for image_base in tqdm(sorted(old_output_dict.keys())):
        old_output = old_output_dict[image_base]
        new_output = modify_output(old_output, new_threshold)
        new_output_dict[image_base] = new_output

    print('dumping...')
    with open(new_output_dict_filename, 'w') as f:
        json.dump(new_output_dict, f)

    print('done dumping')


def apply_post_hoc_threshold_to_grounding_dino(old_threshold, new_threshold):
    old_threshold = float(old_threshold)
    new_threshold = float(new_threshold)

    assert(new_threshold > old_threshold)
    for offset in tqdm(range(DATA_STRIDE)):
        apply_post_hoc_threshold_to_grounding_dino_one_offset(old_threshold, new_threshold, offset)


def usage():
    print('Usage: python apply_post_hoc_threshold_to_grounding_dino.py <old_threshold> <new_threshold>')


if __name__ == '__main__':
    apply_post_hoc_threshold_to_grounding_dino(*(sys.argv[1:]))
