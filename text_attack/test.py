import keras


def index2text(index_input):
    INDEX_FROM = 3
    word_to_id = keras.datasets.imdb.get_word_index()
    # print(word_to_id)
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value: key for key, value in word_to_id.items()}
    # print(' '.join(id_to_word[id] for id in index_input))
    return ' '.join(id_to_word[id] for id in index_input)


def text2index(text_input):
    INDEX_FROM = 3
    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    text_ls = text_input.split(' ')
    return [word_to_id[wd] for wd in text_ls]

max_features = 20000
maxlen = 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_test), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# print(index2text(x_test[0]))
print(x_test[0])
text_test = index2text(x_test[0])
print(text_test)
print(type(text_test))
print(text2index(text_test))

