#!/bin/bash -l

#SBATCH -n 2
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-40:00:00
#SBATCH -C skylake
#SBATCH -J eglvgg19
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=qiang.hu@uni.lu

module purge
module load swenv/default-env/devel
module load system/CUDA numlib/cuDNN

for((i=0;i<25;i++));
do
python -u EGL_VGG19.py
python -u EGL_VGG19_train.py
done
