#!/bin/bash

# $1 : model name
# $2 : model path
# $3 : model base

python -m llava.eval.model_vqa \
    --model-path $2 \
    --model-name coco_syn_llava_lora \
    --model-base $3 \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --max-new-tokens 512

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/eval/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$1.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$1.json

