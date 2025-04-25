import os
import sys
sys.path.append(os.path.abspath('.'))
import shutil
from pathlib import Path
from collections import OrderedDict
from transformers.trainer_utils import get_last_checkpoint
from my_utils.checkpoint import check_best_or_last

if __name__ == '__main__':
    model_names = [
        'train_coco_syn2_adv_ref',
        'train_coco_syn2_adv_ref_combined',
        'train_coco_syn_swap_v1',
        'train_coco_syn_cot_adv_ref',
    ]
    ckpt_root_dir = Path('checkpoint/coco_syn/')
    
    models = OrderedDict({
        model_name : ckpt_root_dir / model_name
        for model_name in model_names
    })
    
    for model_name, ckpt_dir in models.items():
        os.remove(ckpt_dir / 'adapter_model.safetensors')
        last_ckpt_dir = Path(get_last_checkpoint(ckpt_dir))
        # create symbolic link from last checkpoint dir to ckpt_dir
        print('Creating symbolic link for last model')
        os.symlink((last_ckpt_dir / 'adapter_model.safetensors').relative_to(ckpt_dir), ckpt_dir / 'adapter_model.safetensors')