#!/bin/bash -l

# Job name
#SBATCH --job-name=gen_openimage
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
#SBATCH --mem=40gb
# Time limit days-hrs:min:sec
#SBATCH --time 00-20:00:00

# Standard output and error log
#SBATCH --output=/home/yuchenz2/f_verb_alignment/sbatch/log/gen_openimage.out

hostname
echo "job starting"
conda activate alignment

python ../python/gen_data.py openimage openimage_concept.txt glove swav \
  --n_sample 1

echo "job finished!"
