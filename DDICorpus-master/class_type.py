import xml.etree.ElementTree as ET
import string
import os
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense, Embedding, Bidirectional, LSTM
from keras.models import Model, Sequential
from keras import losses
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import json


def remove_punc(text):
    no_punc = "".join([c for c in text if c not in string.punctuation])
    return no_punc


def pre_process(word):
    final_word = word.lower()
    final_word = final_word.strip()
    final_word = remove_punc(final_word)
    return final_word


def create_vocab(filepath):
    tree = ET.parse(file_path)
    root = tree.getroot()
    vocab = []
    x = []
    y_ddi = []
    y_type = []

    for child in root:
        sent = child.attrib['text']
        sentence = sent.split(" ")
        for sub_child in child:
            if sub_child.tag == 'pair' and sub_child.attrib['ddi'] == 'true':
                if 'type' in sub_child.attrib:
                    y_ddi.append(sub_child.attrib['ddi'])
                    y_type.append(sub_child.attrib['type'])
                    x.append(sentence)
                else:
                    y_ddi.append(sub_child.attrib['ddi'])
                    y_type.append('true_other')
                    x.append(sentence)
            elif sub_child.tag == 'pair' and sub_child.attrib['ddi'] == 'false':
                y_ddi.append(sub_child.attrib['ddi'])
                y_type.append('false_other')
                x.append(sentence)
        for word in sentence:
            vocab.append(pre_process(word))
    return vocab, x, y_ddi, y_type


def create_dataset(x, y_ddi, y_type, tokens, ddi_dict, type_dict, max_length):
    final_x = []
    for sent in x:
        dummy_x = []
        for word in sent:
            key = pre_process(word)
            if len(key) != 0:
                dummy_x.append(tokens[key])
        final_x.append(np.asarray(dummy_x))
    final_y_ddi = []
    final_y_type = []
    for i in range(len(y_ddi)):
        final_y_ddi.append(ddi_dict[y_ddi[i]])
        final_y_type.append(type_dict[y_type[i]])
    return final_x, np.asarray(final_y_ddi), np.asarray(final_y_type)


def train(final_x, final_y, reverse_master_tokens, dict_tag, num_epochs, batch_size, embedding_matrix, model_name):
    train_x, test_x, train_y, test_y = train_test_split(final_x, final_y, test_size=0.2)

    input_layer = Input(shape=(train_x.shape[1],))
    model = Embedding(input_dim=len(master_tokens) + 1, output_dim=25, input_length=train_x.shape[1])(input_layer)
    model = Bidirectional(LSTM(units=32, return_sequences=False, recurrent_dropout=0.2))(model)
    model = Dense(128,activation='relu')(model)
    output_layer = Dense(len(dict_tag), activation="softmax")(model)
    model_new = Model(input_layer, output_layer)

    model_new.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model_new.summary()

    history = model_new.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, validation_data=(test_x, test_y))
    model_new.save('%s_model.h5'.format(model_name))

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    for i in range(len(test_x)):
        test_inp = np.asarray([test_x[i]])
        result = model_new.predict(test_inp)
        trans_inp = []
        for word in test_inp[0]:
            if word != 0.0:
                trans_inp.append(reverse_master_tokens[word])
        # print(" ".join(trans_inp))

        reverse_dict_tag = {}
        for key,val in dict_tag.items():
            reverse_dict_tag[val] = key

        with open(model_name+'.txt','a') as f:
            f.write(" ".join(trans_inp))
            f.write("\n")
            f.write(reverse_dict_tag[np.argmax(result[0])])
            f.write("\n")
        f.close()


def read_data(file_name):
    with open(file_name,'r') as f:
        word_vocab = set() # not using list to avoid duplicate entry
        word2vector = {}
        for line in f:
            line_ = line.strip()
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:],dtype=float)
    print("Total Words in DataSet:",len(word_vocab))
    return word_vocab,word2vector


# word_vocab, w2v = read_data('glove.6B.100d.txt')

path = '/Users/vsatpathy/Desktop/DDICorpus-master/APIforDDICorpus/DDICorpus/Train/DrugBank/'
filenames = os.listdir(path)

master_vocab = []
master_x = []
master_y_ddi = []
master_y_type = []
for file in filenames:
    file_path = path + file
    bow, x, y_ddi, y_type = create_vocab(file_path)
    master_vocab.extend(bow)
    master_x.extend(x)
    master_y_ddi.extend(y_ddi)
    master_y_type.extend(y_type)
# print(len(master_x),len(master_y_ddi),len(master_y_type))
# master_vocab = set(master_vocab)

t=Tokenizer(num_words=len(master_vocab))
t.fit_on_texts(master_vocab)
master_tokens = t.word_index

num_words = len(t.word_index) + 1
embedding_matrix = np.zeros((num_words, 25))

for word,i in t.word_index.items():
    try:
        embedding_vector = w2v[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass

reverse_master_tokens = {}
for key,val in master_tokens.items():
    reverse_master_tokens[val] = key

types = sorted(np.unique(master_y_type))
type_dict = {}
for type in types:
    type_dict[type] = len(type_dict)

ddis = sorted(np.unique(master_y_ddi))
ddi_dict = {}
for ddi in ddis:
    ddi_dict[ddi] = len(ddi_dict)

print(type_dict)
print(ddi_dict)

final_x, final_y_ddi, final_y_type = create_dataset(master_x, master_y_ddi, master_y_type, master_tokens, ddi_dict, type_dict, 20)
max_length = 30
final_x = pad_sequences(final_x,maxlen=max_length,padding='post',dtype=float)
# final_y_type = to_categorical(final_y_type,num_classes=len(type_dict))
# final_y_ddi = to_categorical(final_y_ddi,num_classes=len(ddi_dict))
# print(final_x.shape)

train(final_x, final_y_ddi, reverse_master_tokens, ddi_dict, 10, 16, embedding_matrix, 'ddi')
# train(final_x, final_y_type, reverse_master_tokens, type_dict, 30, 16, embedding_matrix, 'type')