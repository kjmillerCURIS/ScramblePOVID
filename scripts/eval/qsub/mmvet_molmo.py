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
        expt_name = f'eval_mmvet_{model_name}'
        
        if model_name == 'molmo':
            model_path = 'allenai/Molmo-7B-D-0924'
        else:
            model_path = str(Path('../trl-new/checkpoint/').resolve() / model_name)
        
        qsub_out = Path(f'playground/data/eval/mm-vet/qsub_outs/{expt_name}.out')
        cmd = ['MKL_THREADING_LAYER=GNU', 'qsub']
        cmd.extend(get_qsub_options(
            qsub_name=f'mmvet_{expt_name}',
            project=PROJECT,
            outfile=qsub_out,
            duration='2:00:00',
            gpu_type='A6000|A100|A40|L40S|L40|RTX6000ada',
        ))
        
        cmd.extend(['bash', 'scripts/eval/bash/mmvet_arg_molmo.sh', model_name, model_path])
        cmd = ' '.join(cmd)
        print(cmd)
        os.system(cmd)