#!/bin/bash -l

#SBATCH -n 5
#SBATCH -N 1
#SBATCH --time=0-40:00:00
#SBATCH -C skylake
#SBATCH -J pwws_yahoo
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=qiang.hu@uni.lu

module purge
module load swenv/default-env/devel
module load system/CUDA numlib/cuDNN

#python -u PWWS_attack.py -model_path ../QC/QCmodels/QC_gru.h5 -data_type qc -result_path results/pwws_attack_results.csv -select_path test1
#python -u PWWS_attack.py -model_path ../QC/QCmodels/BALD_qc_gru_new.h5 -data_type qc -result_path results/pwws_attack_results.csv -select_path test1
#python -u PWWS_attack.py -model_path ../QC/QCmodels/EGL_QC_gru.h5 -data_type qc -result_path results/pwws_attack_results.csv -select_path test1
#python -u PWWS_attack.py -model_path ../QC/QCmodels/entropy_qc_gru_new.h5 -data_type qc -result_path results/pwws_attack_results.csv -select_path test1


python -u PWWS_attack.py -model_path ../Yahoo/models/yahoo_lstm.h5 -data_type yahoo -result_path results/pwws_attack_results_yahoo.csv -select_path test1
