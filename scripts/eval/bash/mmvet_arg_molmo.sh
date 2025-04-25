#!/bin/bash

# $1 : model name
# $2 : model path

python -m llava.eval.model_vqa_molmo \
    --model-path $2 \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$1.jsonl \
    --max_new_tokens 1024
    
mkdir -p ./playground/data/eval/mm-vet/results

python scripts/eval/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$1.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$1.json
