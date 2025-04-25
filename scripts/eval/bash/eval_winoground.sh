#!/bin/bash -l

#$ -V
#$ -N eval_winoground_train_coco_syn2_combined_lora2
#$ -o qsub_outs/eval_winoground_train_coco_syn2_combined_lora2.out
#$ -j y
#$ -P ivc-ml
#$ -pe omp 3
#$ -l h_rt=24:00:00
#$ -l mem_per_core=6G
#$ -m beas
#$ -l gpu=1
#$ -l gpu_memory=40G
#$ -l gpu_type=A6000|A100|A40|L40S|L40|RTX6000ada

# echo "Running eval_ans train_coco_syn2_combined_lora2"
# python eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --model_path checkpoint/coco_syn/train_coco_syn2_combined_lora2 --model_name llava_lora_coco_syn --model_base liuhaotian/llava-v1.5-13b --save_dir expts/winoground/eval_ans_train_coco_syn2_combined_lora2
# echo "Running eval_caption train_coco_syn2_combined_lora2"
# python eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --model_path checkpoint/coco_syn/train_coco_syn2_combined_lora2 --model_name llava_lora_coco_syn --model_base liuhaotian/llava-v1.5-13b --save_dir expts/winoground/eval_caption_train_coco_syn2_combined_lora2 --question "caption:" --answer "{}"

# echo "Running eval_ans train_coco_syn2_lora2"
# python eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --model_path checkpoint/coco_syn/train_coco_syn2_lora2 --model_name llava_lora_coco_syn --model_base liuhaotian/llava-v1.5-13b --save_dir expts/winoground/eval_ans_train_coco_syn2_lora2
# echo "Running eval_caption train_coco_syn2_lora2"
# python eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --model_path checkpoint/coco_syn/train_coco_syn2_lora2 --model_name llava_lora_coco_syn --model_base liuhaotian/llava-v1.5-13b --save_dir expts/winoground/eval_caption_train_coco_syn2_lora2 --question "caption:" --answer "{}"

# echo "Running eval_ans train_sugarcrepe_combined_only_swap_lora2"
# python eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --model_path checkpoint/sugarcrepe/train_sugarcrepe_combined_only_swap_lora2 --model_name llava_lora_sugarcrepe --model_base liuhaotian/llava-v1.5-13b --save_dir expts/winoground/eval_ans_train_sugarcrepe_combined_only_swap_lora2
# echo "Running eval_caption train_sugarcrepe_combined_only_swap_lora2"
# python eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --model_path checkpoint/sugarcrepe/train_sugarcrepe_combined_only_swap_lora2 --model_name llava_lora_sugarcrepe --model_base liuhaotian/llava-v1.5-13b --save_dir expts/winoground/eval_caption_train_sugarcrepe_combined_only_swap_lora2 --question "caption:" --answer "{}"


echo "Running eval_ans train_coco_train_syn_cot_adv_ref_small_batch"
echo "EQBen"
python scripts/eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --dataset eqben_mini --model_name llava_lora_coco_syn --model_path checkpoint/coco_syn/train_coco_train_syn_cot_adv_ref_small_batch --model_base liuhaotian/llava-v1.5-13b --save_dir playground/data/eval/eqben_mini/eval_ans_train_coco_train_syn_cot_adv_ref_small_batch

echo "COLA"
python scripts/eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --dataset cola --model_name llava_lora_coco_syn --model_path checkpoint/coco_syn/train_coco_train_syn_cot_adv_ref_small_batch --model_base liuhaotian/llava-v1.5-13b --save_dir playground/data/eval/cola/eval_ans_train_coco_train_syn_cot_adv_ref_small_batch

echo "Running eval_ans train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000"
echo "Winoground"
python scripts/eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --dataset winoground --model_name llava_lora_sugarcrepe --model_path checkpoint/sugarcrepe/train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000 --model_base liuhaotian/llava-v1.5-13b --save_dir playground/data/eval/winoground/eval_ans_train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000

echo "EQBen"
python scripts/eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --dataset eqben_mini --model_name llava_lora_sugarcrepe --model_path checkpoint/sugarcrepe/train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000 --model_base liuhaotian/llava-v1.5-13b --save_dir playground/data/eval/eqben_mini/eval_ans_train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000

echo "COLA"
python scripts/eval/eval_winoground.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets --dataset cola --model_name llava_lora_sugarcrepe --model_path checkpoint/sugarcrepe/train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000 --model_base liuhaotian/llava-v1.5-13b --save_dir playground/data/eval/cola/eval_ans_train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000
