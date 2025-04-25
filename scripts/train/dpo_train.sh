#!/bin/bash -l

#$ -V
#$ -N second_stage_v2_from_train_coco_syn_cot_no_adv_ref
#$ -o qsub_outs/second_stage_v2_from_train_coco_syn_cot_no_adv_ref.out
#$ -j y
#$ -P ivc-ml
#$ -pe omp 3
#$ -l h_rt=48:00:00
#$ -l mem_per_core=6G
#$ -m beas
#$ -l gpu=1
#$ -l gpu_memory=40G
#$ -l gpu_type=A6000|A100|A40|L40S|L40|RTX6000ada

### -l gpu_type=A6000|A100|A40|L40S|L40|RTX6000ada



# --image_folder_train /projectnb/ivc-ml/array/data/COCO/images/train2017 \
# --image_folder_train /projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe/val2017 \
# --image_folder_train /projectnb/ivc-ml/samarth/datasets/LLaVA-Instruct-150K/images \
    
# --data_path_val data/preference_data/winoground_preference.json \
# --image_folder_val /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/winoground \

# Function to find a free port
find_free_port() {
    port=29500
    while nc -z localhost $port; do
        port=$((port+1))
    done
    echo $port
}

# Get a free port
FREE_PORT=$(find_free_port)
echo "Using port: $FREE_PORT"

# python llava/train/train_dpo.py \
# --model_name_or_path checkpoint/merged/train_coco_syn_feedback_adv_ref/ \
# --model_name_or_path liuhaotian/llava-v1.5-13b \

OLD_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
deepspeed --include localhost:$OLD_CUDA_VISIBLE_DEVICES --master_port $FREE_PORT llava/train/train_dpo.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
    --version v1 \
    --model_name_or_path checkpoint/merged/train_coco_syn_cot_no_adv_ref/ \
    --data_path_train data/preference_data/sugarcrepe_all_preference_combined_only_swap_mix_llava_3000.json \
    --data_path_val data/preference_data/winoground_preference_combined.json \
    --image_folder_train /projectnb/ivc-ml/samarth/datasets/LLaVA-Instruct-150K/images \
    --image_folder_val /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/winoground \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoint/second_stage/second_stage_v2_from_train_coco_syn_cot_no_adv_ref \
    --run_name second_stage_v2_from_train_coco_syn_cot_no_adv_ref \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 200 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --lazy_preprocess True \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --report_to wandb \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model "rewards/accuracies" \
    --greater_is_better True \
    --torch_compile True \
    --label_smoothing 0.1 \
    --avg_logprob True \
    --yesno_weight 0.1
    
    # --rm_yesno_eos True

    # --loss_type "kto_pair"
    # --beta 0.1
    # --loss_type "hinge"

    # --yesno_weight 0.2 \
    # --only_create_config True

python scripts/util/get_last_ckpt.py --model_name second_stage_v2_from_train_coco_syn_cot_no_adv_ref