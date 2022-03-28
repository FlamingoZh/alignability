#!/bin/bash -l

# Job name
#SBATCH --job-name=agg_vg_noun_visual
# Mail events (NONE, BEGIN, END, FAIL, ALL)
###############################################
########## example #SBATCH --mail-type=END,FAIL
##############################################
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eddjou@outlook.com

# Run on a single CPU
#SBATCH --ntasks=1

# Submit job to cpu queue
#SBATCH -p cpu

# Job memory request
#SBATCH --mem=10gb
# Time limit days-hrs:min:sec
#SBATCH --time 00-100:00:00

# Standard output and error log
#SBATCH --output=/home/yuchenz2/f_verb_alignment/sbatch/log/agg_vg_noun_visual.out

hostname
echo "job starting"
conda activate alignment

python ../python/aggregate_exemplars.py vg_noun "../data/dumped_embeddings/vg_noun_concept_least20_swav_glove_20.pkl" visual \
  --n_exemplar_max 20 \
  --n_sample 10

echo "job finished!"
