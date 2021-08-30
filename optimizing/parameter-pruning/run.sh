#!/bin/bash -l

#SBATCH -n 2
#SBATCH -N 1
#SBATCH --time=0-40:00:00
#SBATCH -C skylake
#SBATCH -J pruning
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=qiang.hu@uni.lu

module purge
module load swenv/default-env/devel
module load system/CUDA numlib/cuDNN

python -u pruning.py
