#!/bin/bash -l

#SBATCH -n 3
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-30:00:00
#SBATCH -C skylake
#SBATCH -J wncc_2
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=qiang.hu@uni.lu

module purge
module load swenv/default-env/devel
module load system/CUDA numlib/cuDNN

for((i=0;i<100;i++));
do
python -u EGL_QC.py
python -u EGL_QC_train.py
done
