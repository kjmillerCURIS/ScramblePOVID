#!/bin/bash -l

#$ -P ivc-ml
#$ -l h_rt=23:59:59
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 2
#$ -j y
#$ -m ea

module load miniconda
conda activate imgeneval-crocodile
cd ~/data/ScramblePOVID
python run_grounding_dino_on_full_dataset.py ${THRESHOLD} ${OFFSET}

