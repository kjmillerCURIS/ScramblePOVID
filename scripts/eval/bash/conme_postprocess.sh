#!/bin/bash


CKPT=$1
SPLIT=$2
CHUNKS=$3
# CKPT="molmo"
# SPLIT="replace-att_HUMAN_FILTER"
# CHUNKS=1

output_file=./playground/data/eval/conme/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/conme/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/eval/convert_conme_for_submission.py \
    --annotation-file ./playground/data/eval/conme/${SPLIT}.csv \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/conme/answers/${SPLIT}/${CKPT}.jsonl \
    --output-csv-file ./playground/data/eval/conme/results/${SPLIT}/${CKPT}.csv \
    --ckpt_name $CKPT