from keras.models import load_model
import coremltools
from keras.datasets import cifar10
import glob
import numpy as np
import tensorflow as tf
import argparse
print(tf.__version__)

folder_path = "new_models/RQ1/Yahoo/"
save_folder = "new_models/RQ3/Yahoo/ori_coreML/"
# Image
# csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
#              "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]
csv_names = ["random"]
# IMDB
# csv_names = ["entropy", "BALD", "k_center", "entropy_dropout",  "EGL",
#              "NC", "KMNC", "DSA", "LSA"]
# Yahoo, QC
# csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
#              "NC", "MCP", "KMNC", "DeepGini"]
for csv_name in csv_names:
    print(csv_name)
    model = load_model(folder_path + csv_name + '_lstm_1.h5')
    coreml_model = coremltools.convert(model)
    coreml_model.author = "hq"
    coreml_model.license = 'MIT'
    coreml_model.save(save_folder + csv_name + '_lstm_1.mlmodel')
    print(csv_name + '_2')
    model = load_model(folder_path + csv_name + '_lstm_2' + '.h5')
    coreml_model = coremltools.convert(model)
    coreml_model.author = "hq"
    coreml_model.license = 'MIT'
    coreml_model.save(save_folder + csv_name + '_lstm_2' + '.mlmodel')
    print(csv_name + '_3')
    model = load_model(folder_path + csv_name + '_lstm_3' + '.h5')
    coreml_model = coremltools.convert(model)
    coreml_model.author = "hq"
    coreml_model.license = 'MIT'
    coreml_model.save(save_folder + csv_name + '_lstm_3' + '.mlmodel')

# for csv_name in csv_names:
#     for i in range(20):
#         print(i)
#         model = load_model(folder_path + csv_name + '_' + str(i) + '.h5')
#         coreml_model = coremltools.convert(model)
#         coreml_model.author = "hq"
#         coreml_model.license = 'MIT'
#         coreml_model.save(save_folder + csv_name + '_' + str(i) + '.mlmodel')

#
# for i in range(1, 4):
#     model = load_model(folder_path + 'gru_' + str(i) + '.h5')
#     coreml_model = coremltools.convert(model)
#     coreml_model.author = "hq"
#     coreml_model.license = 'MIT'
#     coreml_model.save(save_folder + 'gru_' + str(i) + '.mlmodel')
# model.summary()
# coreml_model = coremltools.convert(model)
# coreml_model.author = "hq"
# coreml_model.license = 'MIT'
# coreml_model.save("coreML/YahoocoreMLmodelsGRU/yahoo_gru.mlmodel")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", "-model",
#                         type=int,
#                         )
#     # 0-entropy 1-BALD 3-k_center 6-margin 7-entropy_dropout 8-margin_dropout
#     parser.add_argument("--results", "-results",
#                         type=str,
#                         )
#     parser.add_argument("--model", "-model",
#                         type=str,
#                         )
#     parser.add_argument("--model_type", "-model_type",
#                         type=str,
#                         )
#     args = parser.parse_args()
#     metric = args.metric
#     results_path = args.results
#     model_save_path = args.model
#     model_type = args.model_type

