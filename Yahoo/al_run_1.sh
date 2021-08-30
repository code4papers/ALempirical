#!/bin/bash -l

#SBATCH -n 2
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-40:00:00
#SBATCH -C skylake
#SBATCH -J yahoo_kc
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=qiang.hu@uni.lu

module purge
module load swenv/default-env/devel
module load system/CUDA numlib/cuDNN

#python -u active_learning_yahoo.py -metric 0 -results results/entropy_yahoo_lstm.csv -model models/entropy_yahoo_lstm.h5
#python -u active_learning_yahoo.py -metric 1 -results results/BALD_yahoo_lstm.csv -model models/BALD_yahoo_lstm.h5
#python -u active_learning_yahoo.py -metric 2 -results results/entropy_dropout_yahoo_lstm.csv -model models/entropy_dropout_yahoo_lstm.h5
#python -u active_learning_yahoo.py -metric 3 -results results/k_center_yahoo_lstm.csv -model models/k_center_yahoo_lstm.h5
##python -u active_learning_yahoo.py -metric 4 -results results/EGL_yahoo_lstm.csv -model models/EGL_yahoo_lstm.h5
#python -u active_learning_yahoo.py -metric 5 -results results/margin_yahoo_lstm.csv -model models/margin_yahoo_lstm.h5
#python -u active_learning_yahoo.py -metric 6 -results results/margin_dropout_yahoo_lstm.csv -model models/margin_dropout_yahoo_lstm.h5

#python -u active_learning_yahoo.py -metric 0 -results results/entropy_yahoo_gru.csv -model models/entropy_yahoo_gru.h5
#python -u active_learning_yahoo.py -metric 1 -results results/BALD_yahoo_gru.csv -model models/BALD_yahoo_gru.h5
#python -u active_learning_yahoo.py -metric 2 -results results/entropy_dropout_yahoo_gru.csv -model models/entropy_dropout_yahoo_gru.h5
python -u active_learning_yahoo.py -metric 3 -results results/k_center_yahoo_gru.csv -model models/k_center_yahoo_gru.h5
#python -u active_learning_yahoo.py -metric 4 -results results/EGL_yahoo_lstm.csv -model models/EGL_yahoo_lstm.h5
#python -u active_learning_yahoo.py -metric 5 -results results/margin_yahoo_gru.csv -model models/margin_yahoo_gru.h5
#python -u active_learning_yahoo.py -metric 6 -results results/margin_dropout_yahoo_gru.csv -model models/margin_dropout_yahoo_gru.h5
