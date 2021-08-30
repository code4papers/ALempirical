#!/bin/bash -l

#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-40:00:00
#SBATCH -C skylake
#SBATCH -J wncc_2
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=qiang.hu@uni.lu

module purge
module load swenv/default-env/devel
module load system/CUDA numlib/cuDNN

python -u active_learning_QC.py -metric 0 -results results/entropy_qc_gru_new.csv -model QCmodels/entropy_qc_gru_new.h5
python -u active_learning_QC.py -metric 1 -results results/BALD_qc_gru_new.csv -model QCmodels/BALD_qc_gru_new.h5
python -u active_learning_QC.py -metric 2 -results results/entropy_dropout_qc_gru_new.csv -model QCmodels/entropy_dropout_qc_gru_new.h5
python -u active_learning_QC.py -metric 3 -results results/k_center_qc_gru_new.csv -model QCmodels/k_center_qc_gru_new.h5
#python -u active_learning_QC.py -metric 4 -results results/EGL_qc_gru_new.csv -model QCmodels/EGL_qc_gru_new.h5
python -u active_learning_QC.py -metric 5 -results results/margin_qc_gru_new.csv -model QCmodels/margin_qc_gru_new.h5
python -u active_learning_QC.py -metric 6 -results results/margin_dropout_qc_gru_new.csv -model QCmodels/margin_dropout_qc_gru_new.h5
