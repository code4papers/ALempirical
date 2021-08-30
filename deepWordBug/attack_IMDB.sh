#!/bin/bash -l

python -u attack.py -model_path ../IMDB/IMDB_models/imdb_gru_categorical.h5 -dataset imdb -save_path results/dwb_attack_results_imdb.csv -data_select ../IMDB/data/CLEVER_data1.npy

