#!/bin/bash -l

#$ -V
#$ -N second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr3
#$ -o qsub_outs/second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr3.out
#$ -j y
#$ -P ivc-ml
#$ -pe omp 3
#$ -l h_rt=24:00:00
#$ -l mem_per_core=6G
#$ -m beas
#$ -l gpu=1
#$ -l gpu_memory=40G
#$ -l gpu_type=A6000|A100|A40|L40S|L40


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

# --model_name_or_path liuhaotian/llava-v1.5-13b \
# --image_folder /projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe/val2017

OLD_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
deepspeed --include localhost:$OLD_CUDA_VISIBLE_DEVICES --master_port $FREE_PORT llava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path checkpoint/merged/train_coco_syn_cot_adv_ref/ \
    --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 6e-8 \
    --version v1 \
    --data_path data/sft_data/sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000.json \
    --image_folder /projectnb/ivc-ml/samarth/datasets/LLaVA-Instruct-150K/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoint/second_stage/second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr3 \
    --run_name second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr3 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 3e-8 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 3 \
    --report_to wandb \
    --torch_compile True