# coding: utf-8
import os
import numpy as np
from config import config
import copy
import sys
# from read_files import split_imdb_files, split_yahoo_files, split_agnews_files
import spacy
import argparse
import re
from collections import Counter, defaultdict

nlp = spacy.load('en')
parser = argparse.ArgumentParser('named entity recognition')
parser.add_argument('-d', '--dataset',
                    help='Data set',
                    choices=['imdb', 'agnews', 'yahoo'],
                    default='yahoo')

NE_type_dict = {
    'PERSON': defaultdict(int),  # People, including fictional.
    'NORP': defaultdict(int),  # Nationalities or religious or political groups.
    'FAC': defaultdict(int),  # Buildings, airports, highways, bridges, etc.
    'ORG': defaultdict(int),  # Companies, agencies, institutions, etc.
    'GPE': defaultdict(int),  # Countries, cities, states.
    'LOC': defaultdict(int),  # Non-GPE locations, mountain ranges, bodies of water.
    'PRODUCT': defaultdict(int),  # Object, vehicles, foods, etc.(Not services)
    'EVENT': defaultdict(int),  # Named hurricanes, battles, wars, sports events, etc.
    'WORK_OF_ART': defaultdict(int),  # Titles of books, songs, etc.
    'LAW': defaultdict(int),  # Named documents made into laws.
    'LANGUAGE': defaultdict(int),  # Any named language.
    'DATE': defaultdict(int),  # Absolute or relative dates or periods.
    'TIME': defaultdict(int),  # Times smaller than a day.
    'PERCENT': defaultdict(int),  # Percentage, including "%".
    'MONEY': defaultdict(int),  # Monetary values, including unit.
    'QUANTITY': defaultdict(int),  # Measurements, as of weight or distance.
    'ORDINAL': defaultdict(int),  # "first", "second", etc.
    'CARDINAL': defaultdict(int),  # Numerals that do not fall under another type.
}


def recognize_named_entity(texts):
    '''
    Returns all NEs in the input texts and their corresponding types
    '''
    NE_freq_dict = copy.deepcopy(NE_type_dict)

    for text in texts:
        text = text.tolist()
        doc = nlp(text)
        for word in doc.ents:
            NE_freq_dict[word.label_][word.text] += 1
    return NE_freq_dict


def find_adv_NE(D_true, D_other):
    '''
    find NE_adv in D-D_y_true which is defined in the end of section 3.1
    '''
    # adv_NE_list = []
    for type in NE_type_dict.keys():
        # find the most frequent true and other NEs of the same type
        true_NE_list = [NE_tuple[0] for (i, NE_tuple) in enumerate(D_true[type]) if i < 15]
        other_NE_list = [NE_tuple[0] for (i, NE_tuple) in enumerate(D_other[type]) if i < 30]

        for other_NE in other_NE_list:
            if other_NE not in true_NE_list and len(other_NE.split()) == 1:
                # adv_NE_list.append((type, other_NE))
                print("'" + type + "': '" + other_NE + "',")
                with open('./qc.txt', 'a', encoding='utf-8') as f:
                    f.write("'" + type + "': '" + other_NE + "',\n")
                break


class NameEntityList(object):
    # If the original input in IMDB belongs to class 0 (negative)
    imdb_0 = {'PERSON': 'David',
              'NORP': 'Australian',
              'FAC': 'Hound',
              'ORG': 'Ford',
              'GPE': 'India',
              'LOC': 'Atlantic',
              'PRODUCT': 'Highly',
              'EVENT': 'Depression',
              'WORK_OF_ART': 'Casablanca',
              'LAW': 'Constitution',
              'LANGUAGE': 'Portuguese',
              'DATE': '2001',
              'TIME': 'hours',
              'PERCENT': '98%',
              'MONEY': '4',
              'QUANTITY': '70mm',
              'ORDINAL': '5th',
              'CARDINAL': '7',
              }
    # If the original input in IMDB belongs to class 1 (positive)
    imdb_1 = {'PERSON': 'Lee',
              'NORP': 'Christian',
              'FAC': 'Shannon',
              'ORG': 'BAD',
              'GPE': 'Seagal',
              'LOC': 'Malta',
              'PRODUCT': 'Cat',
              'EVENT': 'Hugo',
              'WORK_OF_ART': 'Jaws',
              'LAW': 'RICO',
              'LANGUAGE': 'Sebastian',
              'DATE': 'Friday',
              'TIME': 'minutes',
              'PERCENT': '75%',
              'MONEY': '$',
              'QUANTITY': '9mm',
              'ORDINAL': 'sixth',
              'CARDINAL': 'zero',
              }
    imdb = [imdb_0, imdb_1]

    qc_0 = {'PERSON': 'obama',
            'NORP': 'libyan',
            'FAC': 'broadway',
            'ORG': 'nato',
            'GPE': 'japan',
            'LOC': 'earth',
            'PRODUCT': 'joplin',
            'EVENT': 'watergate',
            'WORK_OF_ART': 'nobel',
            'LAW': 'sub-$10,000',
            'LANGUAGE': 'spanish',
            'DATE': 'march',
            'TIME': 'hours',
            'PERCENT': '6%',
            'QUANTITY': 'gallon',
            'ORDINAL': '3ds',
            'CARDINAL': 'thousands'
            }
    qc_1 = {'PERSON': 'barry',
            'NORP': 'republican',
            'FAC': 'broadway',
            'ORG': 'nba',
            'GPE': 'boston',
            'LOC': 'earth',
            'PRODUCT': 'ericsson',
            'EVENT': 'series',
            'WORK_OF_ART': '10:18',
            'LAW': 'sub-$10,000',
            'LANGUAGE': 'spanish',
            'DATE': 'today',
            'TIME': 'tonight',
            'PERCENT': '6%',
            'MONEY': '100,000',
            'QUANTITY': 'gallon',
            'ORDINAL': 'eighth',
            'CARDINAL': '2'
            }
    qc_2 = {'PERSON': 'barry',
            'NORP': 'french',
            'FAC': 'broadway',
            'ORG': 'nato',
            'GPE': 'china',
            'LOC': 'europe',
            'PRODUCT': 'ericsson',
            'EVENT': 'series',
            'WORK_OF_ART': 'nobel',
            'LAW': 'sub-$10,000',
            'LANGUAGE': 'spanish',
            'DATE': 'today',
            'TIME': 'tonight',
            'PERCENT': '27%',
            'MONEY': '100,000',
            'QUANTITY': '3',
            'ORDINAL': 'seventh',
            'CARDINAL': '2',
            }
    qc_3 = {
        'PERSON': 'obama',
        'NORP': 'libyan',
        'FAC': 'broadway',
        'ORG': 'nato',
        'GPE': 'boston',
        'LOC': 'earth',
        'PRODUCT': 'ericsson',
        'EVENT': 'series',
        'WORK_OF_ART': '10:18',
        'LAW': 'mars',
        'LANGUAGE': 'portuguese',
        'DATE': '2009',
        'TIME': 'night',
        'PERCENT': '27%',
        'MONEY': '100,000',
        'QUANTITY': 'tons',
        'ORDINAL': 'ninth',
        'CARDINAL': 'six',
    }
    qc_4 = {
        'PERSON': 'obama',
        'NORP': 'libyan',
        'FAC': 'broadway',
        'ORG': 'nato',
        'GPE': 'boston',
        'LOC': 'west',
        'PRODUCT': 'ericsson',
        'EVENT': 'series',
        'WORK_OF_ART': 'nobel',
        'LAW': 'sub-$10,000',
        'LANGUAGE': 'english',
        'DATE': 'march',
        'TIME': 'tonight',
        'PERCENT': '6%',
        'MONEY': '500',
        'QUANTITY': 'gallon',
        'ORDINAL': 'eighth',
        'CARDINAL': 'seven',
    }
    qc_5 = {
        'PERSON': 'obama',
        'NORP': 'libyan',
        'FAC': 'mosque',
        'ORG': 'nato',
        'GPE': 'china',
        'LOC': 'europe',
        'PRODUCT': 'ericsson',
        'EVENT': 'olympics',
        'WORK_OF_ART': 'nobel',
        'LAW': 'sub-$10,000',
        'LANGUAGE': 'portuguese',
        'DATE': 'march',
        'TIME': 'hours',
        'PERCENT': '6%',
        'MONEY': '100,000',
        'QUANTITY': 'gallon',
        'ORDINAL': 'ninth',
        'CARDINAL': '2',
    }
    qc_6 = {
        'PERSON': 'barry',
        'NORP': 'libyan',
        'FAC': 'broadway',
        'ORG': 'nato',
        'GPE': 'libya',
        'LOC': 'midwest',
        'PRODUCT': 'ericsson',
        'EVENT': 'olympics',
        'WORK_OF_ART': 'nobel',
        'LAW': 'sub-$10,000',
        'LANGUAGE': 'spanish',
        'DATE': 'march',
        'TIME': 'morning',
        'PERCENT': '6%',
        'MONEY': '100,000',
        'QUANTITY': 'gallon',
        'ORDINAL': 'eighth',
        'CARDINAL': 'seven',
    }
    qc = [qc_0, qc_1, qc_2, qc_3, qc_4, qc_5, qc_6]

    agnews_0 = {'PERSON': 'Williams',
                'NORP': 'European',
                'FAC': 'Olympic',
                'ORG': 'Microsoft',
                'GPE': 'Australia',
                'LOC': 'Earth',
                'PRODUCT': '#',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'PowerBook',
                'LAW': 'Pacers-Pistons',
                'LANGUAGE': 'Chinese',
                'DATE': 'third-quarter',
                'TIME': 'Tonight',
                'MONEY': '#39;t',
                'QUANTITY': '#39;t',
                'ORDINAL': '11th',
                'CARDINAL': '1',
                }
    agnews_1 = {'PERSON': 'Bush',
                'NORP': 'Iraqi',
                'FAC': 'Outlook',
                'ORG': 'Microsoft',
                'GPE': 'Iraq',
                'LOC': 'Asia',
                'PRODUCT': '#',
                'EVENT': 'Series',
                'WORK_OF_ART': 'Nobel',
                'LAW': 'Constitution',
                'LANGUAGE': 'French',
                'DATE': 'third-quarter',
                'TIME': 'hours',
                'MONEY': '39;Keefe',
                'ORDINAL': '2nd',
                'CARDINAL': 'Two',
                }
    agnews_2 = {'PERSON': 'Arafat',
                'NORP': 'Iraqi',
                'FAC': 'Olympic',
                'ORG': 'AFP',
                'GPE': 'Baghdad',
                'LOC': 'Earth',
                'PRODUCT': 'Soyuz',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'PowerBook',
                'LAW': 'Constitution',
                'LANGUAGE': 'Filipino',
                'DATE': 'Sunday',
                'TIME': 'evening',
                'MONEY': '39;m',
                'QUANTITY': '20km',
                'ORDINAL': 'eighth',
                'CARDINAL': '6',
                }
    agnews_3 = {'PERSON': 'Arafat',
                'NORP': 'Iraqi',
                'FAC': 'Olympic',
                'ORG': 'AFP',
                'GPE': 'Iraq',
                'LOC': 'Kashmir',
                'PRODUCT': 'Yukos',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'Gazprom',
                'LAW': 'Pacers-Pistons',
                'LANGUAGE': 'Hebrew',
                'DATE': 'Saturday',
                'TIME': 'overnight',
                'MONEY': '39;m',
                'QUANTITY': '#39;t',
                'ORDINAL': '11th',
                'CARDINAL': '6',
                }
    agnews = [agnews_0, agnews_1, agnews_2, agnews_3]
    yahoo_0 = {'PERSON': 'Fantasy',
               'NORP': 'Russian',
               'FAC': 'Taxation',
               'ORG': 'Congress',
               'GPE': 'U.S.',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Constitution',
               'LANGUAGE': 'Hebrew',
               'DATE': '2004-05',
               'TIME': 'morning',
               'MONEY': '$ale',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Tertiary',
               'CARDINAL': 'three',
               }
    yahoo_1 = {'PERSON': 'Equine',
               'NORP': 'Japanese',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'UK',
               'LOC': 'Sea',
               'PRODUCT': 'RuneScape',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Strap-',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'night',
               'PERCENT': '100%',
               'MONEY': 'five-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Sixth',
               'CARDINAL': '5',
               }
    yahoo_2 = {'PERSON': 'Equine',
               'NORP': 'Canadian',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'California',
               'LOC': 'Atlantic',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'night',
               'PERCENT': '100%',
               'MONEY': 'ten-dollar',
               'QUANTITY': '$ale',
               'ORDINAL': 'Tertiary',
               'CARDINAL': 'two',
               }
    yahoo_3 = {'PERSON': 'Equine',
               'NORP': 'Irish',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'California',
               'LOC': 'Sea',
               'PRODUCT': 'RuneScape',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LAW': 'Strap-',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'tonight',
               'PERCENT': '100%',
               'MONEY': 'five-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Sixth',
               'CARDINAL': '5',
               }
    yahoo_4 = {'PERSON': 'Equine',
               'NORP': 'Irish',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'Canada',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Constitution',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'seconds',
               'PERCENT': '100%',
               'MONEY': 'hundred-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '100',
               }
    yahoo_5 = {'PERSON': 'Equine',
               'NORP': 'English',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'Australia',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LAW': 'Strap-',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'seconds',
               'MONEY': 'hundred-dollar',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '2000',

               }
    yahoo_6 = {'PERSON': 'Fantasy',
               'NORP': 'Islamic',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'California',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'seconds',
               'PERCENT': '100%',
               'MONEY': '$ale',
               'QUANTITY': '$ale',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '100',
               }
    yahoo_7 = {'PERSON': 'Fantasy',
               'NORP': 'Canadian',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'UK',
               'LOC': 'West',
               'PRODUCT': 'Variable',
               'EVENT': 'Watergate',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Constitution',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'tonight',
               'PERCENT': '100%',
               'MONEY': '$ale',
               'QUANTITY': '$ale',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '2000',
               }
    yahoo_8 = {'PERSON': 'Equine',
               'NORP': 'Japanese',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'Chicago',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Strap-',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'night',
               'PERCENT': '100%',
               'QUANTITY': '$ale',
               'ORDINAL': 'Sixth',
               'CARDINAL': '2',

               }
    yahoo_9 = {'PERSON': 'Equine',
               'NORP': 'Chinese',
               'FAC': 'Music',
               'ORG': 'Digital',
               'GPE': 'U.S.',
               'LOC': 'Atlantic',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LAW': 'Constitution',
               'LANGUAGE': 'Spanish',
               'DATE': '1918-1945',
               'TIME': 'night',
               'PERCENT': '100%',
               'MONEY': 'ten-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '5'
               }
    yahoo = [yahoo_0, yahoo_1, yahoo_2, yahoo_3, yahoo_4, yahoo_5, yahoo_6, yahoo_7, yahoo_8, yahoo_9]
    L = {'imdb': imdb, 'agnews': agnews, 'yahoo': yahoo, 'qc': qc}


NE_list = NameEntityList()

def get_tokenizer():
    import pandas as pd
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    max_features = 10000
    max_len = 100
    texts = []
    labels = []
    train_data_path = '../data/train_data_pytorch.csv'
    test_data_path = '../data/test_data_pytorch.csv'
    qc_train = pd.read_csv(train_data_path,
                           names=['num', 'title', 'description', 'category'])
    # print(qc['description'][0])
    for i in range(len(qc_train)):
        texts.append(str(qc_train['description'][i]))
        labels.append(qc_train['category'][i])

    qc_test = pd.read_csv(test_data_path,
                          names=['num', 'title', 'description', 'category'])
    # print(qc['description'][0])
    for i in range(len(qc_test)):
        texts.append(str(qc_test['description'][i]))
        labels.append(qc_test['category'][i])

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_len)
    labels = np.asarray(labels)
    return tokenizer, texts, data, labels


if __name__ == '__main__':
    # args = parser.parse_args()
    # print('dataset:', args.dataset)
    class_num = 7

    # if args.dataset == 'imdb':
    #     train_texts, train_labels, test_texts, test_labels = split_imdb_files()
    #     # get input texts in different classes
    #     pos_texts = train_texts[:12500]
    #     pos_texts.extend(test_texts[:12500])
    #     neg_texts = train_texts[12500:]
    #     neg_texts.extend(test_texts[12500:])
    #     texts = [neg_texts, pos_texts]
    # elif args.dataset == 'agnews':
    #     texts = [[] for i in range(class_num)]
    #     train_texts, train_labels, test_texts, test_labels = split_agnews_files()
    #     for i, label in enumerate(train_labels):
    #         texts[np.argmax(label)].append(train_texts[i])
    #     for i, label in enumerate(test_labels):
    #         texts[np.argmax(label)].append(test_texts[i])
    # elif args.dataset == 'yahoo':
    #     train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
    #     texts = [[] for i in range(class_num)]
    #     for i, label in enumerate(train_labels):
    #         texts[np.argmax(label)].append(train_texts[i])
    #     for i, label in enumerate(test_labels):
    #         texts[np.argmax(label)].append(test_texts[i])
    tokenizer, texts, index_data, labels = get_tokenizer()
    train_indices = np.load("../QC/data/training_indices.npy")
    test_indices = np.load("../QC/data/test_indices.npy")
    texts = np.asarray(texts)
    # print(texts)
    # print(test_indices)
    train_texts = texts[train_indices]
    test_texts = texts[test_indices]
    x_test = index_data[test_indices]
    test_labels = labels[test_indices]
    train_labels = labels[train_indices]
    from keras.utils import to_categorical
    # y_test = to_categorical(y_test, 7)
    texts = [[] for i in range(class_num)]
    for i, label in enumerate(train_labels):
        texts[label].append(train_texts[i])
    for i, label in enumerate(test_labels):
        texts[label].append(test_texts[i])

    D_true_list = []
    for i in range(class_num):
        D_true = recognize_named_entity(texts[i])  # D_true contains the NEs in input texts with the label y_true
        D_true_list.append(D_true)

    for i in range(class_num):
        D_true = copy.deepcopy(D_true_list[i])
        D_other = copy.deepcopy(NE_type_dict)
        for j in range(class_num):
            if i == j:
                continue
            for type in NE_type_dict.keys():
                # combine D_other[type] and D_true_list[j][type]
                for key in D_true_list[j][type].keys():
                    D_other[type][key] += D_true_list[j][type][key]
        for type in NE_type_dict.keys():
            D_other[type] = sorted(D_other[type].items(), key=lambda k_v: k_v[1], reverse=True)
            D_true[type] = sorted(D_true[type].items(), key=lambda k_v: k_v[1], reverse=True)
        print('\nfind adv_NE_list in class', i)
        with open('./{}.txt'.format('qc'), 'a', encoding='utf-8') as f:
            f.write('\nfind adv_NE_list in class' + str(i))
        find_adv_NE(D_true, D_other)
