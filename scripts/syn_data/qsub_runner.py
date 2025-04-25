import os
import sys
sys.path.append(os.path.abspath('.'))
from collections import OrderedDict
from my_utils.qsub import get_qsub_options
from pathlib import Path

PROJECT = 'ivc-ml'
NUM_JOBS = 60

if __name__ == '__main__':
    # rule_types = ['swap_obj', 'swap_att']
    # for rule_type in rule_types:
    # rule_type = 'swap_att'
    for job_idx in [17, 22, 25, 27, 29, 31]:
    # for job_idx in range(16, 30):
        # expt_name = f'gen_neg_caps_llama3_{job_idx}'
        # expt_name = f'gen_neg_single_rule_v2_llama3_{rule_type}_{job_idx}'
        # expt_name = f'gen_neg_single_rule_llama3_{rule_type}_{job_idx}'
        # expt_name = f'gen_neg_cot_llama3_{job_idx}'
        expt_name = f'gen_neg_w_feedback_train_syn_{job_idx}'
        # expt_name = f'test_neg_caps_llama3_{job_idx}'
        # expt_name = f'test_viz_diff_llama3_{job_idx}'
        
        out_dir = Path(f'data/coco_train_syn/coco_caps_neg_feedback')
        # out_dir = Path(f'data/my_sugarcrepe/neg_cot')
        # out_dir = Path(f'data/coco_syn/coco_caps_neg_feedback')
        qsub_out_dir = str(out_dir / 'qsub')
        os.makedirs(qsub_out_dir, exist_ok=True)
        
        cmd = ['qsub']
        cmd.extend(get_qsub_options(
            qsub_name=expt_name,
            project=PROJECT,
            outfile=Path(qsub_out_dir) / f'log_{job_idx}.txt',
            duration='48:00:00',
            gpu_type='A6000|A100|A40|L40S|L40|RTX6000ada',
            gpu_count=1,
            num_workers=3,
        ))
        # cmd.extend(['python', 'scripts/syn_data/gen_neg_single_rule_llama3.py'])
        cmd.extend(['python', 'scripts/syn_data/gen_neg_w_feedback.py'])
        # cmd.extend(['--caption_file', 'playground/dev/sugarcrepe_swap_pos_caps.json'])
        # cmd.extend(['--caption_file', 'playground/dev/coco_25k_caps_train3_cleaned.json'])
        cmd.extend(['--caption_file', 'playground/dev/coco_train_caps_cleaned.json'])
        # cmd.extend(['--rule', rule_type])
        # cmd.extend(['python', 'scripts/syn_data/test_viz_diff_llama3.py'])
        cmd.extend(['--base_seed', '44'])
        cmd.extend(['--cap_start_idx', '0']) # to get the caps strating from some idx
        # cmd.extend(['--num_caps', '20000'])
        cmd.extend(['--num_jobs', str(NUM_JOBS)])
        cmd.extend(['--job_idx', str(job_idx)])
        cmd.extend(['--output_dir', str(out_dir)])
        cmd.extend(['--flush_every', '2'])
        
        # cmd.extend(['--caption_file', f'{out_dir}/caps_{job_idx}.jsonl'])
        # cmd.extend(['--caption_file', f'{out_dir}/combined_caps.json'])
        # cmd.extend(['--output_dir', str(out_dir)])
        # cmd.extend(['--job_idx', str(job_idx)])
        # cmd.extend(['--num_jobs', str(NUM_JOBS)])
        # cmd.extend(['--flush_every', '2'])
        
        cmd = ' '.join(cmd)
        print(cmd)
        os.system(cmd)