#!/bin/bash -l

#$ -P ivc-ml
#$ -l h_rt=5:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate imgeneval-meow
cd ~/data/ScramblePOVID
python generate_hard_negatives_for_scramble.py ${OFFSET}

