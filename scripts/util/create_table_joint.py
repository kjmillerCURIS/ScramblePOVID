import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pandas as pd
from itertools import product
import numpy as np

if __name__ == '__main__':
    model_names = [
        # 'llava-v1.5-13b',
        # 'train_sugarcrepe_only_swap_combined_w_diff_contexts_lora2',
        # 'train_coco_syn2_lora2',
        # 'train_coco_syn2_combined_lora2',
        # 'train_coco_syn2_adv_ref_lora2',
        # 'train_coco_syn2_adv_ref_combined_lora2',
        # 'train_coco_syn_swap_v1_short',
        # 'train_coco_syn_swap_v2_short',
        # 'train_coco_syn_combined_lora2_downwt_0.1',
        # 'train_coco_syn_combined_lora2_downwt_0.2',
        # 'train_coco_syn_swap_v1_len_filtered'
        # 'train_coco_syn2_adv_ref',
        # 'train_coco_syn2_adv_ref_combined',
        # 'train_coco_syn_swap_v1',
        # 'train_coco_syn_cot_adv_ref',
        
        # 'train_my_sugarcrepe_swap_v1',
        # 'train_my_sugarcrepe_swap_v1_combined',
        # 'train_my_sugarcrepe_neg_cot_filled',
        # 'train_my_sugarcrepe_neg_cot_filled_combined',
        # 'train_sugarcrepe_only_swap_run2',
        # 'train_sugarcrepe_only_swap_combined_run2',
        
        # 'train_coco_train_syn_swap',
        # 'train_coco_train_syn_feedback_adv_ref_small_batch',
        # 'train_sugarcrepe_only_swap_combined_run3_deepspeed'
        # 'train_my_sugarcrepe_swap_v1_adv_ref_filled_combined',
        
        # 'train_sugarcrepe_combined_only_swap_labsmooth_0.2',
        # 'train_sugarcrepe_combined_only_swap_labsmooth_0.1',
        # 'train_sugarcrepe_combined_only_swap_hinge_loss',
        # 'train_sugarcrepe_combined_only_swap_sppo_hard_beta_0.3',
        # 'train_sugarcrepe_combined_only_swap_kto',
        # 'train_sugarcrepe_combined_only_swap_sppo_hard_beta_0.1',
        # 'train_sugarcrepe_combined_only_swap_labsmooth_0.1_logprob_avg',
        # 'train_sugarcrepe_combined_only_swap_labsmooth_0.1_rmyesno_eos',
        # 'train_sugarcrepe_combined_w_explanation_only_swap_labsmooth_0.1',
        
        # 'train_coco_syn_cot_adv_ref_combined_w_explanation_labsmooth_0.1_avg_logprob',
        # 'train_coco_syn_cot_adv_ref_combined_labsmooth_0.1_avg_logprob',
        # 'train_coco_syn_cot_adv_ref_combined_w_explanation_labsmooth_0.1'
        
        # 'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile',
        # 'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile_tr_longer',
        # 'train_coco_syn_cot_adv_ref_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1',
        # 'train_coco_syn_cot_adv_ref_labsmooth_0.1_avg_logprob',
        
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        # 'second_stage_coco_syn_cot_adv_ref',
        # 'train_coco_syn_cot_adv_ref_rand1000_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        # 'second_stage_coco_syn_cot_adv_ref_rand1000_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        
        # 'train_coco_train_syn_cot_adv_ref',
        # 'train_coco_train_syn_cot_adv_ref_high_lr',
        # 'train_coco_train_syn_cot_adv_ref_small_batch',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_1000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_3000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_5000',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_1000',
        # 'train_coco_syn_cot_adv_ref_llava_caps',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer_llava_caps',
        # 'train_coco_syn_feedback_adv_ref',
        # 'train_coco_train_syn_feedback_adv_ref_high_lr',
        
        # 'second_stage_from_train_coco_train_syn_cot_adv_ref_high_lr',
        # 'second_stage_from_train_coco_syn_feedback_adv_ref',
        # 'second_stage_from_train_coco_train_syn_feedback_adv_ref_high_lr',
        
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
        
        # 'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000',
        # 'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr2',
        # 'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr3',
        # 'train_coco_train_syn_cot_adv_ref_lr2',
        # 'train_coco_train_syn_cot_adv_ref_lr3',
        
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_lr2',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_lr3',
        
        # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_small_batch',
        # 'second_stage_v3_from_train_coco_train_syn_cot_adv_ref_small_batch',
        # 'second_stage_v3_from_train_coco_syn_feedback_adv_ref',
        
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_high_lr'
        
        # 'molmo',
        # 'molmo_train_coco_syn_cot_adv_ref_1epoch',
        # 'molmo_train_coco_syn_cot_adv_ref_lora2_1epoch',
        # 'molmo_train_coco_syn_cot_adv_ref_lora2',
        # 'molmo_train_coco_syn_cot_adv_ref_lora2_lr2',
        # 'molmo_train_coco_syn_cot_adv_ref_lora2_lr3',
        # 'molmo_train_coco_syn_cot_adv_ref_lora2_5epoch',
        # 'molmo_train_coco_syn_cot_adv_ref_llava_caps_lora2',
        
        # 'molmo_train_coco_train_syn_cot_adv_ref_lora2',
        # 'molmo_train_coco_train_syn_cot_adv_ref_llava_caps_lora2',
        
        'llama3',
        'llama3_train_coco_syn_cot_adv_ref_1epoch',
        # 'llama3_train_coco_syn_cot_adv_ref_lora2_1epoch',
        # 'llama3_train_coco_syn_cot_adv_ref_lora2',
        # 'llama3_train_coco_syn_cot_adv_ref_llava_caps',
        'llama3_train_coco_train_syn_cot_adv_ref_1epoch',
    ]
    
    ## COMPARISON OF STAGES
    # model_names = [
    #     'llava-v1.5-13b',
    #     'train_coco_train_syn_cot_adv_ref_small_batch',
    #     'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
    #     'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_small_batch',
    # ]
    # eval_types = ['ans', 'caption']

    ## COMPARISON OF AMT OF TRAINING DATA
    # model_names = [
    #     'llava-v1.5-13b',
    #     'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_10pct',
    #     'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_25pct',
    #     'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_50pct',
    #     'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_small_batch',
    # ]
    
    ### COMPARISON USING 10K SAMPLES FOR SWAP VS COT
    # model_names = [
    #     'llava-v1.5-13b',
    #     'train_coco_train_syn_swap_10k_random',
    #     'train_coco_train_syn_swap_10k_most_plausible',
    #     'train_coco_train_syn_cot_adv_ref_10k_random',
    #     'train_coco_train_syn_cot_adv_ref_10k_most_plausible',
    # ]
    
    
    ## COMPARISON OF AMT OF TRAINING DATA : Stage 1
    # model_names = [
    #     'llava-v1.5-13b',
    #     'train_coco_train_syn_cot_adv_ref_10pct',
    #     'train_coco_train_syn_cot_adv_ref_25pct',
    #     'train_coco_train_syn_cot_adv_ref_50pct',
    #     'train_coco_train_syn_cot_adv_ref_small_batch',
    # ]
    # eval_types = ['ans', 'caption']
    
    ## COMPARISON W/O FILTRATION
    # model_names = [
    #     'train_coco_syn_cot_adv_ref',
    #     'train_coco_syn_cot_no_adv_ref',
    #     # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_3000', # 'second_stage_v2_from_train_coco_syn_cot_adv_ref',
    #     # 'second_stage_v2_from_train_coco_syn_cot_no_adv_ref',
    # ]

    # Create a list of tuples for the index
    # index = list(product(model_names, eval_types))
    index = model_names

    # Initialize a dictionary to hold the data
    data = {
        ('Winoground', 'text'): [],
        ('Winoground', 'image'): [], 
        ('Winoground', 'group'): [],
        ('EqBen', 'text'): [],
        ('EqBen', 'image'): [], 
        ('EqBen', 'group'): [],
        ('COLA', 'text'): [],
        ('COLA', 'image'): [], 
        ('COLA', 'group'): [],
        ('ConMe', 'replace-att'): [],
        ('ConMe', 'replace-obj'): [],
        ('ConMe', 'replace-rel'): [],
        ('ConMe', 'total'): [],
        ('SEED-Bench', 'total'): [],
        ('MM-Vet', 'final'): [],
        ('MM-Vet', 'total'): [],
        ('MM-Vet', 'std'): [],
        ('MM-Vet', '95pct_conf'): [],
    }

    # Populate the dictionary with data from the JSON files
    for model_name in index:
        #### WINOGROUND ####
        file_path = f'playground/data/eval/winoground/eval_ans_{model_name}/results.json'
        if not os.path.exists(file_path):
            print(f'Winoground results for {model_name} not found')
            data[('Winoground', 'text')].append(None)
            data[('Winoground', 'image')].append(None)
            data[('Winoground', 'group')].append(None)
        else:
            with open(file_path, 'r') as f:
                results = json.load(f)
                data[('Winoground', 'text')].append(results['all']['text']*100.)
                data[('Winoground', 'image')].append(results['all']['image']*100.)
                data[('Winoground', 'group')].append(results['all']['group']*100.)
        
        
        #### EQBEN ####
        file_path = f'playground/data/eval/eqben_mini/eval_ans_{model_name}/results.json'
        if not os.path.exists(file_path):
            print(f'EqBen results for {model_name} not found')
            data[('EqBen', 'text')].append(None)
            data[('EqBen', 'image')].append(None)
            data[('EqBen', 'group')].append(None)
        else:
            with open(file_path, 'r') as f:
                results = json.load(f)
                data[('EqBen', 'text')].append(results['all']['text']*100.)
                data[('EqBen', 'image')].append(results['all']['image']*100.)
                data[('EqBen', 'group')].append(results['all']['group']*100.)
        
        #### COLA ####
        file_path = f'playground/data/eval/cola/eval_ans_{model_name}/results.json'
        if not os.path.exists(file_path):
            print(f'COLA results for {model_name} not found')
            data[('COLA', 'text')].append(None)
            data[('COLA', 'image')].append(None)
            data[('COLA', 'group')].append(None)
        else:
            with open(file_path, 'r') as f:
                results = json.load(f)
                data[('COLA', 'text')].append(results['all']['text']*100.)
                data[('COLA', 'image')].append(results['all']['image']*100.)
                data[('COLA', 'group')].append(results['all']['group']*100.)
            
        #### CONME ####
        curr_conme_total = 0
        curr_conme_total_num = 0
        for split in ['replace-att', 'replace-obj', 'replace-rel']:
            file_path = f'playground/data/eval/conme/results/{split}/{model_name}.csv'
            if not os.path.exists(file_path):
                print(f'ConMe results for {model_name} not found')
                data[('ConMe', split)].append(None)
            else:
                df = pd.read_csv(file_path)
                data[('ConMe', split)].append(df['total'].values[0])
                curr_conme_total += df['total'].values[0]
                curr_conme_total_num += 1
        data[('ConMe', 'total')].append(curr_conme_total/curr_conme_total_num if curr_conme_total_num > 0 else None)
            
        #### SEED-Bench ####
        file_path = f'playground/data/eval/seed_bench/results-image/{model_name}.csv'
        if not os.path.exists(file_path):
            print(f'SEED-Bench results for {model_name} not found')
            data[('SEED-Bench', 'total')].append(None)
        else:
            df = pd.read_csv(file_path)
            data[('SEED-Bench', 'total')].append(df['total'].mean())
        
        #### MM-Vet ####
        file_paths = [f'playground/data/eval/mm-vet/gradio_out/{model_name}_gpt-4-32k-0613-cap-score-1runs.csv']
        for i in range(5):
            file_paths.append(f'playground/data/eval/mm-vet/gradio_out/{model_name}_eval-run-{i}_gpt-4-32k-0613-cap-score-1runs.csv')
        
        all_mmvet_data = []
        for i, file_path in enumerate(file_paths):
            if not os.path.exists(file_path):
                # print(f'MM-Vet results for {model_name} missing run {i}')
                # all_mmvet_data.append(None)
                pass
            else:
                df = pd.read_csv(file_path)
                all_mmvet_data.append(df)
        if len(all_mmvet_data) < 5:
            print(f'MM-Vet results for {model_name} missing runs') 
        
        try:
            all_mmvet_data = pd.concat(all_mmvet_data)
            mmvet_mean = all_mmvet_data['total'].mean()
        except:
            mmvet_mean = None
        
        try:
            mmvet_std = all_mmvet_data['total'].std()
        except:
            mmvet_std = None
        data[('MM-Vet', 'total')].append(mmvet_mean)
        data[('MM-Vet', 'std')].append(mmvet_std)
        if mmvet_std is not None:
            conf_interval = (1.96*mmvet_std)/np.sqrt(len(all_mmvet_data))
            data[('MM-Vet', '95pct_conf')].append(conf_interval)
            data[('MM-Vet', 'final')].append(f'{mmvet_mean:.1f} $\pm$ {conf_interval:.1f}')
        else:
            data[('MM-Vet', '95pct_conf')].append(None)
            data[('MM-Vet', 'final')].append(None)
            
    df = pd.DataFrame(data, index=index)
    df.to_excel('playground/data/eval/joint_tables/llama3_new.xlsx', float_format='%.2f')
    