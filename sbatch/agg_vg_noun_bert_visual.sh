#!/bin/bash -l

# Job name
#SBATCH --job-name=agg_vg_noun_bert_visual
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
#SBATCH --output=/home/yuchenz2/f_verb_alignment/sbatch/log/agg_vg_noun_bert_visual.out

hostname
echo "job starting"
conda activate alignment

python ../python/aggregate_exemplars.py vg_noun_bert "../data/dumped_embeddings_replication/vg_noun_least20_ll_swav_bert_20.pkl" visual \
  --n_exemplar_max 20 \
  --n_sample 100

echo "job finished!"