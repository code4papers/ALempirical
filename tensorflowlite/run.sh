#!/bin/bash -l


python -u ../../tensorflowlite/tflite_prediction.py -model_path ../../new_models/RQ3/QC/tf8bit/ -save_path ../../new_results/RQ3/QC/quantization/tflite.csv -data_type qc



