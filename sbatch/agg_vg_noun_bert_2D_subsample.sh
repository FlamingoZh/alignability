#!/bin/bash -l

# Job name
#SBATCH --job-name=agg_vg_noun_bert_2D_subsample_2nd
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
#SBATCH --output=/home/yuchenz2/f_verb_alignment/sbatch/log/agg_vg_noun_bert_2D_subsample_2nd.out

hostname
echo "job starting"
conda activate alignment

python ../python/aggregate_exemplars_2D_subsample.py vg_noun_bert "../data/dumped_embeddings_replication/vg_noun_least20_ll_2nd_swav_bert_20.pkl" visual_language_subsample \
  --n_l_exemplar_max 8 \
  --n_v_exemplar_max 8 \
  --n_sample 500

echo "job finished!"
