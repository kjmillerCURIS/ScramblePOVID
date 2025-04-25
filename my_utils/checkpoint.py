import shutil
from transformers.trainer_utils import get_last_checkpoint

import os
from pathlib import Path


def check_best_or_last(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir_name = ckpt_dir.name
    
    if '_sft_' in ckpt_dir_name: # NOTE: currently there is no eval for sft models
        return 'last'
    
    best_model_dir = ckpt_dir / 'best_model'
    last_ckpt_dir = Path(get_last_checkpoint(ckpt_dir))

    try:
        link_path = Path(os.readlink(ckpt_dir / 'adapter_model.safetensors'))
        # if link is relative then it is relative to ckpt_dir
        if not link_path.is_absolute():
            link_path = ckpt_dir / link_path
    except Exception as e:
        # TODO : make sure default is best model
        return 'best [default]'

    if link_path.parent.resolve() == best_model_dir.resolve():
        return 'best'
    elif link_path.parent.resolve() == last_ckpt_dir.resolve():
        return 'last'
    else:
        print('Last ckpt dir: ', last_ckpt_dir.resolve())
        print('Best model dir: ', best_model_dir.resolve())
        print('Link path parent: ', link_path.parent.resolve())
        raise Exception(f'Unknown symbolic link: {link_path}')


def get_best_ckpt(ckpt_dir):
    best_or_last = check_best_or_last(ckpt_dir)
    if best_or_last.startswith('best'):
        print('Found best model, skipping')
        return False

    # remove the last model symbolic link
    assert best_or_last == 'last'
    print('Removing symbolic link for last model')
    os.remove(ckpt_dir / 'adapter_model.safetensors')
    # best_or_last == 'last'
    best_model_dir = ckpt_dir / 'best_model'
    # create symbolic link from best model dir to ckpt_dir
    print('Creating symbolic link for best model')
    os.symlink((best_model_dir / 'adapter_model.safetensors').relative_to(ckpt_dir), ckpt_dir / 'adapter_model.safetensors')
    return True


def get_last_ckpt(ckpt_dir):
    best_or_last = check_best_or_last(ckpt_dir)
    if best_or_last == 'last':
        print('Found last model, skipping')
        return False

    best_model_dir = ckpt_dir / 'best_model'
    if best_or_last == 'best [default]':
        os.makedirs(best_model_dir)
        print('Moving adapter_model.safetensors to best_model')
        shutil.move(ckpt_dir / 'adapter_model.safetensors', best_model_dir / 'adapter_model.safetensors')
    elif best_or_last == 'best':
        # best model exists in dir, need to remove the symbolic link
        print('Removing symbolic link for best model')
        os.remove(ckpt_dir / 'adapter_model.safetensors')

    last_ckpt_dir = Path(get_last_checkpoint(ckpt_dir))
    # create symbolic link from last checkpoint dir to ckpt_dir
    print('Creating symbolic link for last model')
    os.symlink((last_ckpt_dir / 'adapter_model.safetensors').relative_to(ckpt_dir), ckpt_dir / 'adapter_model.safetensors')
    return True