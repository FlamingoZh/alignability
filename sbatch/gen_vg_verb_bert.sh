#!/bin/bash -l

# Job name
#SBATCH --job-name=gen_vg_verb_bert
# Mail events (NONE, BEGIN, END, FAIL, ALL)
###############################################
########## example #SBATCH --mail-type=END,FAIL
##############################################
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eddjou@outlook.com

# Run on a single CPU
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# Submit job to cpu queue
#SBATCH -p gpu

# Job memory request
#SBATCH --mem=40gb
# Time limit days-hrs:min:sec
#SBATCH --time 00-100:00:00

# Standard output and error log
#SBATCH --output=/home/yuchenz2/f_verb_alignment/sbatch/log/gen_vg_verb_bert.out

hostname
echo "job starting"
conda activate alignment

python ../python/gen_data.py vg_verb vg_verb_concept_least20.txt bert swav \
  --n_sample 20 \
  --cuda

echo "job finished!"
