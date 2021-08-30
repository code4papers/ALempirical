from coremltools.models.neural_network import quantization_utils
import coremltools

# dirs = ["BALD_imdb_gru.mlmodel", "EGL_imdb_gru.mlmodel", "k_center_imdb_gru.mlmodel"]
# dirs_ml = ["BALD_imdb_gru_2bit.mlmodel", "EGL_imdb_gru_2bit.mlmodel", "k_center_imdb_gru_2bit.mlmodel"]
folder_path = "../new_models/RQ3/Yahoo/ori_coreML/"
save_folder = "../new_models/RQ3/Yahoo/"
# Image
# csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
#              "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]

# csv_names = ["margin_dropout", "EGL",
#              "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]
# csv_names = ["random"]

# IMDB
csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "entropy_dropout",  "EGL",
             "NC", "KMNC"]
# Yahoo, QC
# csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
#              "NC", "MCP", "KMNC", "DeepGini"]

bits = [2, 4, 8]
# bits = [2, 4, 8]
for csv_name in csv_names:
    model_fp32 = coremltools.models.MLModel(folder_path + csv_name + '_lstm_1.mlmodel')
    for bit in bits:
        model_2bit = quantization_utils.quantize_weights(model_fp32, nbits=bit)
        model_2bit.save(save_folder + str(bit) + 'bit/' + csv_name + '_lstm_1.mlmodel')

    model_fp32 = coremltools.models.MLModel(folder_path + csv_name + '_lstm_2.mlmodel')
    for bit in bits:
        model_2bit = quantization_utils.quantize_weights(model_fp32, nbits=bit)
        model_2bit.save(save_folder + str(bit) + 'bit/' + csv_name + '_lstm_2.mlmodel')

    model_fp32 = coremltools.models.MLModel(folder_path + csv_name + '_lstm_3.mlmodel')
    for bit in bits:
        model_2bit = quantization_utils.quantize_weights(model_fp32, nbits=bit)
        model_2bit.save(save_folder + str(bit) + 'bit/' + csv_name + '_lstm_3.mlmodel')

