#!/bin/bash -l

#SBATCH -n 10
#SBATCH -N 2
#SBATCH --time=0-30:00:00
#SBATCH -C skylake
#SBATCH -J wncc_2
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=qiang.hu@uni.lu

module purge
module load swenv/default-env/devel
module load system/CUDA numlib/cuDNN

python -u get_NE_list.py
