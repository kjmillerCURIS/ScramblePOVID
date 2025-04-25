import os
import sys
sys.path.append(os.path.abspath('.'))
from my_utils.qsub import get_qsub_options
from collections import OrderedDict

PROJECT = 'ivc-ml'

if __name__ == "__main__":
    models = OrderedDict({
        # 'train_coco_train_syn_cot_adv_ref_lr2' : 'checkpoint/coco_syn/train_coco_train_syn_cot_adv_ref_lr2',
        # 'train_coco_train_syn_cot_adv_ref_lr3' : 'checkpoint/coco_syn/train_coco_train_syn_cot_adv_ref_lr3',
        'train_coco_train_syn_feedback_adv_ref_small_batch' : 'checkpoint/coco_syn/train_coco_train_syn_feedback_adv_ref_small_batch',
    })
    for model_name, model_path in models.items():
        cmd = ['qsub']
        cmd.extend(get_qsub_options(
            qsub_name=f'merge_lora_weights_{model_name}',
            project=PROJECT,
            outfile=f'qsub_outs/merge_lora_weights_{model_name}.out',
            duration='1:00:00',
            gpu_type='A6000|A100|A40|L40S|L40|RTX6000ada',
        ))
        cmd.extend(['python', 'scripts/util/merge_lora_weights.py'])
        cmd.extend(['--model-path', model_path])
        cmd.extend(['--model-base', 'liuhaotian/llava-v1.5-13b'])
        cmd.extend(['--model-name', f'llava_lora_{model_name}'])
        cmd.extend(['--save-model-path', f'checkpoint/merged/{model_name}'])
        cmd = ' '.join(cmd)
        print(cmd)
        os.system(cmd)