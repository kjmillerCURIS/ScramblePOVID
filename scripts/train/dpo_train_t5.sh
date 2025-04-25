#!/bin/bash -l

#$ -V
#$ -N t5_train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile
#$ -o qsub_outs/t5_train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile.out
#$ -j y
#$ -P ivc-ml
#$ -pe omp 3
#$ -l h_rt=48:00:00
#$ -l mem_per_core=6G
#$ -m beas
#$ -l gpu=1
#$ -l gpu_memory=40G
#$ -l gpu_type=A6000|A100|A40|L40S|L40|RTX6000ada



# --image_folder_train /projectnb/ivc-ml/array/data/COCO/images/train2017 \
# --image_folder_train /projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe/val2017 \
    
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
OLD_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
deepspeed --include localhost:$OLD_CUDA_VISIBLE_DEVICES --master_port $FREE_PORT llava/train/train_dpo_t5.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
    --model_name_or_path zhiqiulin/clip-flant5-xxl \
    --version t5_v1 \
    --data_path_train data/preference_data/sugarcrepe_all_preference_combined_only_swap.json \
    --data_path_val data/preference_data/winoground_preference_combined.json \
    --image_folder_train /projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe/val2017 \
    --image_folder_val /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/winoground \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoint/sugarcrepe/t5_train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile \
    --run_name t5_train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8\
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
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
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --report_to wandb \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model "rewards/accuracies" \
    --greater_is_better True \
    --label_smoothing 0.1 \
    --avg_logprob True \
    --torch_compile True \
    --yesno_weight 0.1 \
    
    # --rm_yesno_eos True

    # --loss_type "kto_pair"
    # --beta 0.1
    # --loss_type "hinge"

    # --yesno_weight 0.2 \
    # --only_create_config True

python scripts/util/get_last_ckpt.py --model_name t5_train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile