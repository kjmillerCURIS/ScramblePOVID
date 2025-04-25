import os
import sys
sys.path.append(os.path.abspath('.'))

if __name__ == '__main__':
    model_names = [
        'molmo_train_coco_syn_cot_adv_ref_llava_caps_lora2',
        # 'molmo_train_coco_train_syn_cot_adv_ref_lora2',
        # 'molmo_train_coco_train_syn_cot_adv_ref_llava_caps_lora2'
    ]
    
    tasks = [
        'cola',
        'winoground',
        'eqben_mini',
        'seedbench',
        'mmvet',
    ]
    
    for model_name in model_names:
        print(f'####### RUNNING {model_name} #######')
        for task in tasks:
            print(f'TASK : {task}')
            if task in ['cola', 'winoground', 'eqben_mini']:
                os.system(f'python scripts/eval/qsub/gpt_score_molmo.py --eval_dataset {task} --model_name {model_name}')
            else:
                os.system(f'python scripts/eval/qsub/{task}_molmo.py --model_name {model_name}')
