#! /bin/bash

echo 'Creating table for COLA'
python scripts/util/create_table_cola.py;
cp playground/data/eval/cola/eval_results.xlsx playground/data/eval/src_tables/src_eval_results_cola.xlsx

echo 'Creating table for Winoground'
python scripts/util/create_table_winoground.py;
cp playground/data/eval/winoground/eval_results.xlsx playground/data/eval/src_tables/src_eval_results_winoground.xlsx

echo 'Creating table for Eqben Mini'
python scripts/util/create_table_eqben_mini.py;
cp playground/data/eval/eqben_mini/eval_results.xlsx playground/data/eval/src_tables/src_eval_results_eqben_mini.xlsx

echo 'Creating table for SeedBench'
python scripts/util/create_table_seedbench.py;
cp playground/data/eval/seed_bench/eval_results_seedbench_image.xlsx playground/data/eval/src_tables/src_eval_results_seedbench_image.xlsx

echo 'Creating table for MMVET'
python scripts/util/create_table_mmvet.py;
cp playground/data/eval/mm-vet/eval_results_mmvet.xlsx playground/data/eval/src_tables/src_eval_results_mmvet.xlsx
