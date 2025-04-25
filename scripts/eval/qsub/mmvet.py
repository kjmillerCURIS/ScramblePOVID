import os
import sys
sys.path.append(os.path.abspath('.'))
from collections import OrderedDict
from my_utils.qsub import get_qsub_options
from pathlib import Path
from my_utils.checkpoint import check_best_or_last
import re
import argparse
PROJECT = 'ivc-ml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    
    if not args.model_name:
        model_names = [
            # 'train_coco_syn_cot_adv_ref',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
            
            # 'train_coco_train_syn_cot_adv_ref_high_lr',
            # 'second_stage_from_train_coco_train_syn_cot_adv_ref_high_lr',
            # 'train_coco_train_syn_cot_adv_ref_small_batch',
            
            # 'train_coco_syn_feedback_adv_ref',
            # 'second_stage_from_train_coco_syn_feedback_adv_ref',
            
            # 'train_coco_train_syn_feedback_adv_ref_high_lr',
            # 'second_stage_from_train_coco_train_syn_feedback_adv_ref_high_lr',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_5000',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_3000',
            
            # 'train_coco_syn_cot_adv_ref_w_sugarcrepe',
            # 'train_coco_syn_cot_adv_ref_w_sugarcrepe_long_tr',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.3_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.5_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000_tr_longer',
            
            # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_high_lr',
            # 'second_stage_v2_from_train_coco_syn_feedback_adv_ref',
            
            'train_coco_train_syn_swap',
            # 'second_stage_v2_from_train_coco_syn_swap_v1',
            # 'second_stage_v2_from_train_coco_train_syn_swap',
            # 'second_stage_v2_from_train_coco_syn2_adv_ref',
            # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_small_batch',
        ]
    else:
        model_names = [args.model_name]
        
    ckpt_root_map = OrderedDict({
        'train_my_sugarcrepe': 'checkpoint/my_sugarcrepe',
        'train_sugarcrepe': 'checkpoint/sugarcrepe',
        'train_coco': 'checkpoint/coco_syn',
        'second_stage': 'checkpoint/second_stage',
    })
    
    model_base_map = OrderedDict({
        'train' : 'liuhaotian/llava-v1.5-13b',
        'second_stage' : 'checkpoint/merged/train_coco_syn_cot_adv_ref/',
    })
    
    models = OrderedDict()
    for model_name in model_names:
        for start_str, ckpt_root_dir in ckpt_root_map.items():
            if model_name.startswith(start_str):
                ckpt_root_dir = Path(ckpt_root_dir)
                break
        models[model_name] = ckpt_root_dir / model_name
    
    for model in models:
        if re.match(r'^second_stage.*_from_', model):
            base_name = model.split('_from_')[1]
            model_base = f'checkpoint/merged/{base_name}'
        else:
            for base_prefix in model_base_map:
                if model.startswith(base_prefix):
                    model_base = model_base_map[base_prefix]
                    break
        
        expt_name = f'eval_mmvet_{model}'
        
        if check_best_or_last(models[model]).startswith('best'):
            print('Found best model for model: ', model)
            expt_name += '_best_ckpt'
        
        model_path = str(models[model])
        qsub_out = Path(f'playground/data/eval/mm-vet/qsub_outs/{expt_name}.out')
        cmd = ['qsub']
        cmd.extend(get_qsub_options(
            qsub_name=f'mmvet_{expt_name}',
            project=PROJECT,
            outfile=qsub_out,
            duration='3:00:00',
            gpu_type='A6000|A100|A40|L40S|L40|RTX6000ada',
        ))
        
        cmd.extend(['bash', 'scripts/eval/bash/mmvet_arg.sh', model, model_path, model_base])
        cmd = ' '.join(cmd)
        print(cmd)
        os.system(cmd)