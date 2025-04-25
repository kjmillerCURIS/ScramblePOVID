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
        model_names = [
            # 'molmo',
            # 'molmo_train_coco_syn_cot_adv_ref_1epoch',
            # 'molmo_train_coco_syn_cot_adv_ref_lora2_1epoch',
            # 'molmo_train_coco_syn_cot_adv_ref_lora2',
            # 'molmo_train_coco_syn_cot_adv_ref_lora2_lr2',
            # 'molmo_train_coco_syn_cot_adv_ref_lora2_lr3',
            # 'molmo_train_coco_syn_cot_adv_ref_lora2_5epoch',
            'molmo_train_coco_syn_cot_adv_ref_llava_caps_lora2',
            
            # 'molmo_train_coco_train_syn_cot_adv_ref_lora2',
            # 'molmo_train_coco_train_syn_cot_adv_ref_llava_caps_lora2',
        ]
    else:
        model_names = [args.model_name]
    
    for model_name in model_names:
        if model_name == 'molmo':
            model_path = 'allenai/Molmo-7B-D-0924'
        else:
            model_path = str(Path('../trl-new/checkpoint/').resolve() / model_name)
            
        for job_idx in range(NUM_JOBS):
            # ckpt_name = 'train_sugarcrepe_combined_only_swap_lora2'
            expt_name = f'seedbench_image_eval_{model_name}_{job_idx}'
            
            qsub_out_dir = Path('./playground/data/eval/seed_bench/') / 'qsub_outs'
            os.makedirs(qsub_out_dir, exist_ok=True)
            
            cmd = ['MKL_THREADING_LAYER=GNU', 'qsub']
            cmd.extend(get_qsub_options(
                qsub_name=expt_name,
                project=PROJECT,
                outfile=Path(qsub_out_dir) / f'{expt_name}_jobid_{job_idx}.txt',
                duration='4:00:00',
                gpu_type='A6000|A100|A40|L40S|L40|RTX6000ada',
                gpu_count=1,
                num_workers=3,
            ))
            
            cmd.extend(['python', '-m', 'llava.eval.model_vqa_molmo'])
            cmd.extend(['--model-path', str(model_path)])
            
            cmd.extend(['--question-file', './playground/data/eval/seed_bench/llava-seed-bench-image.jsonl'])
            cmd.extend(['--image-folder', './playground/data/eval/seed_bench'])
            cmd.extend(['--answers-file', f'./playground/data/eval/seed_bench/answers-image/{model_name}/{NUM_JOBS}_{job_idx}.jsonl'])
            cmd.extend(['--num-chunks', str(NUM_JOBS)])
            cmd.extend(['--chunk-idx', str(job_idx)])
            cmd.extend(['--max_new_tokens', '32'])
            
            cmd = ' '.join(cmd)
            print(cmd)
            os.system(cmd)
        
        
        