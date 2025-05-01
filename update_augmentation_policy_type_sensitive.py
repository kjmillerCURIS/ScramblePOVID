import os
import sys
import numpy as np
from tqdm import tqdm


#suggest making easy_percentile=10, hard_percentile=90, overhard_percentile=97
#FIXME: right now this uses the top few percentile to rule out "overhard" examples (where we suspect the meaning is basically the same)
#consider doing this by other ways, such as using an LLM, and remember to incorporate that into this function (something like result['is_overhard'] bool flag would do the trick)
def update_augmentation_policy_type_sensitive(params, results, augmentation_policy):
    assert(params['type_sensitive'])
    vals = [result['negative_cossim'] - result['positive_cossim'] for result in results]
    easy_threshold = np.percentile(vals, params['easy_percentile'])
    hard_threshold = np.percentile(vals, params['hard_percentile'])
    overhard_threshold = np.percentile(vals, params['overhard_percentile'])
    numerators = {aug_type : 0 for aug_type in sorted(augmentation_policy['mixing_weights'].keys())}
    denominators = {aug_type : 0 for aug_type in sorted(augmentation_policy['mixing_weights'].keys())}
    for result in results:
        aug_type = result['aug_type']
        val = result['negative_cossim'] - result['positive_cossim']
        is_easy = int(val < easy_threshold)
        is_hard = int(val >= hard_threshold and val < overhard_threshold)
        is_overhard = int(val >= overhard_threshold)
        numerators[aug_type] += is_hard
        if params['prob_type'] == 'hard_vs_easy':
            denominators[aug_type] += is_hard + is_easy
        elif params['prob_type'] == 'hard_vs_easy_vs_overhard':
            denominators[aug_type] += is_hard + is_easy + is_overhard
        elif params['prob_type'] == 'hard_vs_other':
            denominators[aug_type] += 1

    mixing_weights = {aug_type : numerators[aug_type] / max(denominators[aug_type], 1) for aug_type in sorted(augmentation_policy['mixing_weights'].keys())}
    total = np.sum([mixing_weights[aug_type] for aug_type in sorted(mixing_weights.keys())])
    mixing_weights = {aug_type : mixing_weights / total for aug_type in sorted(mixing_weights.keys())}
    augmentation_policy['mixing_weights'] = mixing_weights
    return augmentation_policy
