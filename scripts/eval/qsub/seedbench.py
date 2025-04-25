import os
import sys
sys.path.append(os.path.abspath('.'))
from my_utils.qsub import get_qsub_options

from pathlib import Path
from collections import OrderedDict
import re
import argparse

PROJECT = 'ivc-ml'
NUM_JOBS = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    
    if not args.model_name:
        all_ckpts = [
            # 'llava-v1.5-13b',
            # 'train_coco_syn_combined_lora2_incompl',
            # 'train_coco_syn_lora2',
            # 'train_coco_syn2_adv_ref_lora2',
            # 'train_coco_syn2_adv_ref_combined_lora2',
            # 'train_coco_syn_swap_v1',
            # 'train_coco_syn_swap_v2',
            # 'train_coco_syn_cot_adv_ref',
            # 'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile_tr_longer',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
            # 'second_stage_coco_syn_cot_adv_ref',
            
            # 'train_coco_syn_cot_adv_ref_rand1000_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
            # 'second_stage_coco_syn_cot_adv_ref_rand1000_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_1000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_5000',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_1000',
            # 'train_coco_syn_cot_adv_ref_llava_caps',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer_llava_caps',
            # 'train_coco_train_syn_feedback_adv_ref_high_lr',
            # 'train_coco_train_syn_cot_adv_ref_high_lr',
            # 'train_coco_syn_feedback_adv_ref',
            
            # 'train_coco_syn_cot_adv_ref_w_sugarcrepe',
            # 'train_coco_syn_cot_adv_ref_w_sugarcrepe_long_tr',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_3000',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.3_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.5_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000_tr_longer',
            
            # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_high_lr',
            # 'second_stage_v2_from_train_coco_syn_feedback_adv_ref',
            
            # 'second_stage_v2_from_train_coco_syn_swap_v1',
            # 'second_stage_v2_from_train_coco_train_syn_swap',
            # 'second_stage_v2_from_train_coco_syn2_adv_ref',
            # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_small_batch',
            # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_high_lr',
            
            # 'train_coco_train_syn_cot_adv_ref_small_batch',
            
            # 'train_coco_train_syn_cot_adv_ref_10pct',
            # 'train_coco_train_syn_cot_adv_ref_25pct',
        ]
    else:
        all_ckpts = [args.model_name]
        
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
    
    for ckpt_name in all_ckpts:
        if re.match(r'^second_stage.*_from_', ckpt_name):
            base_name = ckpt_name.split('_from_')[1]
            model_base = f'checkpoint/merged/{base_name}'
        else:
            for base_prefix in model_base_map:
                if ckpt_name.startswith(base_prefix):
                    model_base = model_base_map[base_prefix]
                    break
            
        for job_idx in range(NUM_JOBS):
        # for job_idx in [0]:
            # ckpt_name = 'train_sugarcrepe_combined_only_swap_lora2'
            expt_name = f'seedbench_image_eval_{ckpt_name}_{job_idx}'
            
            for start_str, ckpt_root_dir in ckpt_root_map.items():
                if ckpt_name.startswith(start_str):
                    ckpt_root_dir = Path(ckpt_root_dir)
                    break
            model_path = ckpt_root_dir / ckpt_name
            
            qsub_out_dir = Path('./playground/data/eval/seed_bench/') / 'qsub_outs'
            os.makedirs(qsub_out_dir, exist_ok=True)
            
            cmd = ['qsub']
            cmd.extend(get_qsub_options(
                qsub_name=expt_name,
                project=PROJECT,
                outfile=Path(qsub_out_dir) / f'{expt_name}_jobid_{job_idx}.txt',
                duration='4:00:00',
                gpu_type='A6000|A100|A40|L40S|L40|RTX6000ada',
                gpu_count=1,
                num_workers=3,
            ))
            
            cmd.extend(['python', '-m', 'llava.eval.model_vqa_loader'])
            # cmd.extend(['--model-path', 'liuhaotian/llava-v1.5-13b'])
            cmd.extend(['--model-path', str(model_path)])
            cmd.extend(['--model-name', 'llava_lora_coco_syn'])
            cmd.extend(['--model-base', model_base])
            
            cmd.extend(['--model-name', 'llava_lora_coco_syn'])
            # cmd.extend(['--model-name', 'llava_v1.5_13b'])
            # cmd.extend(['--model-base', 'liuhaotian/llava-v1.5-13b'])
            
            cmd.extend(['--question-file', './playground/data/eval/seed_bench/llava-seed-bench-image.jsonl'])
            cmd.extend(['--image-folder', './playground/data/eval/seed_bench'])
            cmd.extend(['--answers-file', f'./playground/data/eval/seed_bench/answers-image/{ckpt_name}/{NUM_JOBS}_{job_idx}.jsonl'])
            cmd.extend(['--num-chunks', str(NUM_JOBS)])
            cmd.extend(['--chunk-idx', str(job_idx)])
            cmd.extend(['--temperature', '0'])
            cmd.extend(['--conv-mode', 'vicuna_v1'])
            
            cmd = ' '.join(cmd)
            print(cmd)
            os.system(cmd)
        
        
        