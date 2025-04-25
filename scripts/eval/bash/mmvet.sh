#!/bin/bash

CKPT_NAME="train_coco_syn_combined_lora2_incompl"

python -m llava.eval.model_vqa \
    --model-path checkpoint/coco_syn/${CKPT_NAME} \
    --model-name sugarcrepe_llava_lora \
    --model-base liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/eval/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${CKPT_NAME}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${CKPT_NAME}.json

