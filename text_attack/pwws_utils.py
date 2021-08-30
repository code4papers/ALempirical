
from keras.preprocessing.sequence import pad_sequences


def text_to_vector(text, tokenizer, dataset):
    if dataset == 'qc':
        sequences = tokenizer.texts_to_sequences([text])
        data = pad_sequences(sequences, maxlen=100)
    elif dataset == 'yahoo':
        sequences = tokenizer.texts_to_sequences([text])
        data = pad_sequences(sequences, maxlen=1000)
    elif dataset == 'imdb':
        # print(text)
        text_ls = text.split(' ')
        # print(text_ls)
        # print(text_ls)
        text_len = len(text_ls)
        if text_len < 200:
            start_index = 200 - text_len
            data = [0 for i in range(200 - text_len)]
            data[-1] = 1
        else:
            data = [1]
        # for i in range(start_index, 200):
        for wd in text_ls:
            token = tokenizer.get(wd, 2)
            data.append(token)
            # if token < 20000:
            #     data.append(token)
            # else:
            #     data.append(2)
            # token = tokenizer.get(text_ls[i], 2)
            # if token < 20000:
            #     data.append(token)
            # else:
            #     data.append(2)
        data = [data]

    return data


def index2text(index_input, tokenizer):
    id_to_word = {value: key for key, value in tokenizer.items()}

    index_input = [x for x in index_input if x != 0 and x != 1 and x != 2]
    # print(' '.join(id_to_word[id] for id in index_input))
    return ' '.join(id_to_word[id] for id in index_input)
