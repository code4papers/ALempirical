from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper, TensorFlowModelWrapper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import textattack
import tensorflow as tf
from textattack.datasets import HuggingFaceDataset
from tensorflow.keras.models import load_model
import keras
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# original_model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
original_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
original_model = load_model("../IMDB/IMDB_models/imdb_lstm_glove.h5")
model = TensorFlowModelWrapper(original_model)
# print(model(['I hate you so much', 'I love you']))
attack = TextFoolerJin2019.build(model)
dataset = HuggingFaceDataset("rotten_tomatoes", None, "test", shuffle=True)
max_features = 20000
maxlen = 200
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

results_iterable = attack.attack_dataset(x_val, indices=range(10))
for result in results_iterable:
  print(result)
  # print(result.__str__(color_method='ansi'))
# print(results_iterable)
