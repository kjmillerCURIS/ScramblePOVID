import os
import sys
import json
from tqdm import tqdm
from run_grounding_dino_on_subset import BASE_DIR, get_output_dict_filename, DETECTOR_TYPE
from display_image_with_caption import display_image_with_caption
from compute_groundingdino_stats import categorize


def assertions_on_parser_results(parser_results):
    for label in sorted(parser_results.keys()):
        quant, count = parser_results[label]
        assert(quant in ['exactly', 'at least'])
        assert(count >= 1)
        assert(count >= 2 or quant == 'exactly')


def check_thorough_one(dino_count, parser_quant, parser_count):
    if parser_quant == 'exactly':
        return dino_count == parser_count
    elif parser_quant == 'at least':
        return dino_count >= parser_count
    else:
        assert(False)


def make_decision(output, strategy):
        dino_counts = {}
        parser_results = output['parser_results']
        dino_labels = output['detection_results']['text_labels']
        for label in dino_labels:
            if label not in dino_counts:
                dino_counts[label] = 0

            dino_counts[label] += 1

        for label in sorted(parser_results.keys()):
            if label not in dino_counts:
                dino_counts[label] = 0

        assertions_on_parser_results(parser_results)
        if strategy == 'check_presence':
            is_accepted = all([dino_counts[label] >= 1 for label in sorted(parser_results.keys())])
        elif strategy == 'check_singular_and_plural':
            dino_categories = [categorize(dino_counts[label]) for label in sorted(parser_results.keys())]
            parser_categories = [categorize(parser_results[label][1]) for label in sorted(parser_results.keys())]
            is_accepted = all([dc == pc for dc, pc in zip(dino_categories, parser_categories)])
        elif strategy == 'check_thorough':
            is_accepted = all([check_thorough_one(dino_counts[label], parser_results[label][0], parser_results[label][1]) for label in sorted(parser_results.keys())])
        else:
            assert(False)

        return is_accepted


def run_groundingdino_simulation(threshold, strategy):
    threshold = float(threshold)

    assert(strategy in ['check_presence', 'check_singular_and_plural', 'check_thorough'])
    dino_dict_filename = get_output_dict_filename(threshold)
    with open(dino_dict_filename, 'r') as f:
        dino_dict = json.load(f)

    accepted = []
    rejected = []
    for image_base in tqdm(sorted(dino_dict.keys())):
        is_accepted = make_decision(dino_dict[image_base], strategy)
        if is_accepted:
            accepted.append(image_base)
        else:
            rejected.append(image_base)

    print('%d accepted, %d rejected'%(len(accepted), len(rejected)))
    for image_list, vis_dir_base in zip([accepted, rejected], ['accepted', 'rejected']):
        vis_dir = os.path.join(BASE_DIR, 'subset_simulations', '%s_subset_simulation_threshold%s_%s'%(DETECTOR_TYPE, str(threshold), strategy), vis_dir_base)
        for image_base in tqdm(image_list):
            display_image_with_caption(os.path.join(BASE_DIR, 'OpenImages/images/train', image_base), dino_dict[image_base]['caption'], vis_dir)


def usage():
    print('Usage: python run_groundingdino_simulation.py <threshold> <strategy>')


if __name__ == '__main__':
    run_groundingdino_simulation(*(sys.argv[1:]))
