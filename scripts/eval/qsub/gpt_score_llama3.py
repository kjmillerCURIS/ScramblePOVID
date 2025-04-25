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
    parser.add_argument('--eval_dataset', type=str, default=None)
    args = parser.parse_args()
    
    if not args.model_name:
        model_names = [
            'llama3',
            'llama3_train_coco_syn_cot_adv_ref_1epoch',
            'llama3_train_coco_syn_cot_adv_ref_lora2_1epoch',
            'llama3_train_coco_syn_cot_adv_ref_lora2',
        ]
    else:
        model_names = [args.model_name]
    
    if not args.eval_dataset:
        eval_datasets = ['cola', 'winoground', 'eqben_mini']
    else:
        eval_datasets = [args.eval_dataset]
    
    for eval_dataset in eval_datasets:
        for eval_type in ['ans']:
            for model_name in model_names:
                expt_name = f'eval_{eval_type}_{model_name}'
                
                if model_name == 'llama3':
                    model_path = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
                else:
                    model_path = str(Path('../trl-new/checkpoint/').resolve() / model_name)
                
                save_dir = str(Path(f'playground/data/eval/{eval_dataset}/{expt_name}'))
                cmd = ['MKL_THREADING_LAYER=GNU', 'qsub']
                cmd.extend(get_qsub_options(
                    qsub_name=f'{eval_dataset}_{expt_name}',
                    project=PROJECT,
                    outfile=Path(save_dir) / f'qsub/log.txt',
                    duration='1:00:00',
                    gpu_type='A6000|A100|A40|L40S|L40|RTX6000ada',
                ))
                cmd.extend(['python', 'scripts/eval/eval_winoground.py'])
                cmd.extend(['--root_dir', '/projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets'])
                cmd.extend(['--dataset', eval_dataset])
                cmd.extend(['--model_type', 'llama3'])
                cmd.extend(['--model_path', model_path])
                cmd.extend(['--save_dir', save_dir])
                if eval_type == 'caption':
                    cmd.extend(['--question', '"caption:"'])
                    cmd.extend(['--answer', '"{}"'])
                
                cmd = ' '.join(cmd)
                print(cmd)
                os.system(cmd)