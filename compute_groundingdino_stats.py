import os
import sys
import json
import numpy as np
from tqdm import tqdm
from run_grounding_dino_on_subset import BASE_DIR, DETECTOR_TYPE, get_output_dict_filename


def initialize_event_counter():
    event_flagger = {}
    event_flagger['extraneous_dino_labels'] = 0
    event_flagger['false_presence'] = 0
    event_flagger['false_absence'] = 0
    event_flagger['false_plurality'] = 0
    event_flagger['false_singularity'] = 0
    event_flagger['correct_by_count'] = 0
    event_flagger['correct_by_presence_vs_absence'] = 0
    event_flagger['correct_by_plurality_vs_singularity_vs_absence'] = 0
    return event_flagger


def categorize(count):
    if count == 0:
        return 'absent', 'absent'
    elif count == 1:
        return 'present', 'singular'
    else:
        return 'present', 'plural'


def do_counts_match_exactly(dino_count, manual_count):
    return (dino_count == manual_count or (manual_count == 100 and dino_count >= 2))


def print_one_stat(description, event, event_counter, num_images):
    print('%.1f%% (%d/%d) images %s'%(100 * event_counter[event] / num_images, event_counter[event], num_images, description))


def compute_groundingdino_stats(threshold):
    threshold = float(threshold)

    with open(get_output_dict_filename(threshold), 'r') as f:
        groundingdino_dict = json.load(f)

    detector_type = {'GroundingDINO' : 'groundingdino', 'YOLO-World' : 'yoloworld'}[DETECTOR_TYPE]
    with open(os.path.join(BASE_DIR, 'groundingdino_subset_manual_counts.json'), 'r') as f:
        manual_count_dict = json.load(f)

    event_counter = initialize_event_counter()
    num_images = 0
    for k in sorted(manual_count_dict.keys()):
        if k not in groundingdino_dict:
            print('!!')
            continue

        num_images += 1
        event_flagger = initialize_event_counter()
        event_flagger['correct_by_count'] = 1
        event_flagger['correct_by_presence_vs_absence'] = 1
        event_flagger['correct_by_plurality_vs_singularity_vs_absence'] = 1
        manual_counts = manual_count_dict[k]
        dino_counts = {}
        for label in groundingdino_dict[k]['detection_results']['text_labels']:
            if label not in dino_counts:
                dino_counts[label] = 0

            dino_counts[label] += 1

        for label in sorted(manual_counts.keys()):
            if label not in dino_counts:
                dino_counts[label] = 0

        if any([label not in manual_counts for label in sorted(dino_counts.keys())]):
            event_flagger['extraneous_dino_labels'] = 1

        for label in sorted(manual_counts.keys()):
            dino_binary, dino_trinary = categorize(dino_counts[label])
            manual_binary, manual_trinary = categorize(manual_counts[label])
            if dino_binary == 'present' and manual_binary == 'absent':
                event_flagger['false_presence'] = 1
            if dino_binary == 'absent' and manual_binary == 'present':
                event_flagger['false_absence'] = 1
            if dino_trinary == 'plural' and manual_trinary == 'singular':
                event_flagger['false_plurality'] = 1
            if dino_trinary == 'singular' and manual_trinary == 'plural':
                event_flagger['false_singularity'] = 1
            if not do_counts_match_exactly(dino_counts[label], manual_counts[label]):
                event_flagger['correct_by_count'] = 0
            if dino_binary != manual_binary:
                event_flagger['correct_by_presence_vs_absence'] = 0
            if dino_trinary != manual_trinary:
                event_flagger['correct_by_plurality_vs_singularity_vs_absence'] = 0

        for event in sorted(event_flagger.keys()):
            event_counter[event] += event_flagger[event]

    print('THRESHOLD = %s'%(str(threshold)))
    print_one_stat('are completely correct', 'correct_by_count', event_counter, num_images)
    print_one_stat('are correct in terms of presence vs absence', 'correct_by_presence_vs_absence', event_counter, num_images)
    print_one_stat('are correct in terms of singular vs plural vs absent', 'correct_by_plurality_vs_singularity_vs_absence', event_counter, num_images)
    print_one_stat('have at least one false presence (actually is absent, %s thinks is present)'%(detector_type), 'false_presence', event_counter, num_images)
    print_one_stat('have at least one false absence (actually is present, %s thinks is absent)'%(detector_type), 'false_absence', event_counter, num_images)
    print_one_stat('have at least one false plurality (actually is singular, %s thinks is plural)'%(detector_type), 'false_plurality', event_counter, num_images)
    print_one_stat('have at least one false singularity (actually is plural, %s thinks is singular)'%(detector_type), 'false_singularity', event_counter, num_images)
    print_one_stat('have at least one false absence (actually is present, %s thinks is absent)'%(detector_type), 'false_absence', event_counter, num_images)
    #print_one_stat('have at least one extraneous label from dino (weird library quirk, not counted against correctness)', 'extraneous_dino_labels', event_counter, num_images)


def usage():
    print('Usage: python compute_groundingdino_stats.py <threshold>')


if __name__ == '__main__':
    compute_groundingdino_stats(*(sys.argv[1:]))
