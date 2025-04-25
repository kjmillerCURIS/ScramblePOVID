#!/bin/bash
echo "Running eval_ans llava-v1.5-13b"
python eval/eval_sugarcrepe.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe --model_name llava-v1.513b --model_path liuhaotian/llava-v1.5-13b --save_dir expts/sugarcrepe/eval_ans_llava-v1.5-13b
echo "Running eval_caption llava-v1.5-13b"
python eval/eval_sugarcrepe.py --root_dir /projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe --model_name llava-v1.513b --model_path liuhaotian/llava-v1.5-13b --save_dir expts/sugarcrepe/eval_caption_llava-v1.5-13b --question "caption:" --answer "{}"
