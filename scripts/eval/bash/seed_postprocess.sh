#!/bin/bash

# CHUNKS=$2
CHUNKS=5

CKPT=$1
# CKPT="train_sugarcrepe_combined_only_swap_lora2"

output_file=./playground/data/eval/seed_bench/answers-image/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers-image/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/eval/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench-image.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers-image_upload/${CKPT}.jsonl \
    --output-csv-file ./playground/data/eval/seed_bench/results-image/${CKPT}.csv \
    --ckpt_name $CKPT