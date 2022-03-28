#!/bin/bash -l

# Job name
#SBATCH --job-name=agg_mit_bert_2D
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
#SBATCH --time 00-200:00:00

# Standard output and error log
#SBATCH --output=/home/yuchenz2/f_verb_alignment/sbatch/log/agg_mit_bert_2D.out

hostname
echo "job starting"
conda activate alignment

python ../python/aggregate_exemplars_2D.py mit_bert "../data/dumped_embeddings/mit_concept_swav_bert_20.pkl" visual_language \
  --n_l_exemplar_max 20 \
  --n_v_exemplar_max 20 \
  --n_sample 1

echo "job finished!"
