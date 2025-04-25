import os
import sys
sys.path.append(os.path.abspath('.'))

from my_utils.checkpoint import get_last_ckpt
from pathlib import Path
from collections import OrderedDict
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    
    if args.model_name is not None:
        model_names = [args.model_name]
    else:
        model_names = [
            # 'train_coco_syn2_adv_ref',
            # 'train_coco_syn2_adv_ref_combined',
            # 'train_coco_syn_swap_v1',
            # 'train_coco_syn_cot_adv_ref',
            # 'train_my_sugarcrepe_swap_v1',
            # 'train_my_sugarcrepe_swap_v1_combined',
            # 'train_my_sugarcrepe_neg_cot_filled',
            # 'train_my_sugarcrepe_neg_cot_filled_combined',
            # 'train_sugarcrepe_only_swap_run2',
            # 'train_sugarcrepe_only_swap_combined_run2',
            
            # 'train_coco_train_syn_swap',
            # 'train_sugarcrepe_only_swap_combined_run3_deepspeed'
            # 'train_my_sugarcrepe_swap_v1_adv_ref_filled_combined',
            
            # 'train_sugarcrepe_combined_only_swap_labsmooth_0.2',
            # 'train_sugarcrepe_combined_only_swap_labsmooth_0.1',
            # 'train_sugarcrepe_combined_only_swap_hinge_loss',
            # 'train_sugarcrepe_combined_only_swap_sppo_hard_beta_0.3',
            # 'train_sugarcrepe_combined_only_swap_kto',
            
            # 'train_coco_syn_cot_adv_ref_combined_w_explanation_labsmooth_0.1_avg_logprob',
            # 'train_coco_syn_cot_adv_ref_combined_labsmooth_0.1_avg_logprob',
            # 'train_coco_syn_cot_adv_ref_combined_w_explanation_labsmooth_0.1',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
            
            'train_coco_train_syn_cot_adv_ref_10pct',
            'train_coco_train_syn_cot_adv_ref_25pct',
            'train_coco_train_syn_cot_adv_ref_50pct',
        ]
    # ckpt_root_dir = Path('checkpoint/my_sugarcrepe/')
    
    ckpt_root_map = OrderedDict({
        'train_my_sugarcrepe': 'checkpoint/my_sugarcrepe',
        'train_sugarcrepe': 'checkpoint/sugarcrepe',
        'train_coco': 'checkpoint/coco_syn',
        'second_stage': 'checkpoint/second_stage',
    })
    models = OrderedDict()
    for model_name in model_names:
        for start_str, ckpt_root_dir in ckpt_root_map.items():
            if model_name.startswith(start_str):
                ckpt_root_dir = Path(ckpt_root_dir)
                break
        models[model_name] = ckpt_root_dir / model_name
    
    for model_name, ckpt_dir in models.items():
        print('Model name: ', model_name)
        get_last_ckpt(ckpt_dir)
        