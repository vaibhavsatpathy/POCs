import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense, Embedding, Bidirectional, LSTM
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def remove_punc(text):
    no_punc = "".join([c for c in text if c not in string.punctuation])
    return no_punc


def pre_process(word):
    final_word = word.lower()
    final_word = final_word.strip()
    final_word = remove_punc(final_word)
    return final_word


def create_vocab(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    drug_id = {}
    vocab = []
    X = []
    Y_ddi = []
    Y_type = []
    for child in root:
        for sub_child in child:
            temp_x = []
            temp_y = []
            temp_y_type = []
            if sub_child.tag == 'entity':
                drug_id[sub_child.attrib['id']] = sub_child.attrib['text']
                vocab.append(sub_child.attrib['text'])
            if sub_child.tag == 'pair' and sub_child.attrib['ddi'] == 'true':
                if 'type' in sub_child.attrib:
                    # print(drug_id[sub_child.attrib['e1']], drug_id[sub_child.attrib['e2']], sub_child.attrib['ddi'], sub_child.attrib['type'])
                    temp_x.append(drug_id[sub_child.attrib['e1']])
                    temp_x.append(drug_id[sub_child.attrib['e2']])
                    temp_y.append(sub_child.attrib['ddi'])
                    temp_y_type.append(sub_child.attrib['type'])
                else:
                    # print(drug_id[sub_child.attrib['e1']], drug_id[sub_child.attrib['e2']], sub_child.attrib['ddi'], 'other')
                    temp_x.append(drug_id[sub_child.attrib['e1']])
                    temp_x.append(drug_id[sub_child.attrib['e2']])
                    temp_y.append(sub_child.attrib['ddi'])
                    temp_y_type.append('other')
            elif sub_child.tag == 'pair' and sub_child.attrib['ddi'] == 'false':
                # print(drug_id[sub_child.attrib['e1']], drug_id[sub_child.attrib['e2']], sub_child.attrib['ddi'])
                temp_x.append(drug_id[sub_child.attrib['e1']])
                temp_x.append(drug_id[sub_child.attrib['e2']])
                temp_y.append(sub_child.attrib['ddi'])
                temp_y_type.append('false_other')

            if len(temp_x) != 0 and len(temp_y) != 0:
                temp_x = np.asarray(temp_x)
                temp_y = np.asarray(temp_y)
                temp_y_type = np.asarray(temp_y_type)
                X.append(temp_x)
                Y_ddi.extend(temp_y)
                Y_type.extend(temp_y_type)
    # print(drug_id)
    X = np.asarray(X)
    Y_ddi = np.asarray(Y_ddi)
    Y_type = np.asarray(Y_type)
    return vocab, X, Y_ddi, Y_type


def create_dataset(data_x, tokens):
    master_data_vocab = {}
    for word_pair in data_x:
        master_data_vocab[pre_process(word_pair[0])] = len(master_data_vocab)
        master_data_vocab[pre_process(word_pair[1])] = len(master_data_vocab)
    # print(len(set(master_data_vocab)))

    final_x = []
    for word_pair in data_x:
        # print(pre_process(word_pair[1]))
        dummy_x = []
        dummy_x.append(master_data_vocab[pre_process(word_pair[0])])
        dummy_x.append(master_data_vocab[pre_process(word_pair[1])])
        dummy_x = np.asarray(dummy_x)
        final_x.append(dummy_x)
    final_x = np.asarray(final_x)
    return final_x, master_data_vocab


def infer(input_text, tokens):
    dummy_text = input_text.split(" ")
    updated_tok = []
    for word in dummy_text:
        updated_tok.append(tokens[pre_process(word)])
    return updated_tok


def train_ddi(final_x, final_y_ddi, master_tokens):
    train_x, test_x, train_y, test_y = train_test_split(final_x, final_y_ddi, test_size=0.2)

    input_layer = Input(shape=(train_x.shape[1],))
    # model=Embedding(input_dim=len(master_data_vocab)+1,output_dim=32,input_length=train_x.shape[1])(input_layer)
    # model = Bidirectional(LSTM(units = 32,return_sequences=True, recurrent_dropout=0.2)) (model)
    model = Dense(16, activation='relu')(input_layer)
    output_layer = Dense(len(np.unique(final_y_ddi)), activation="softmax")(model)
    model_new = Model(input_layer, output_layer)

    model_new.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model_new.summary()

    history = model_new.fit(train_x, train_y, epochs=20, batch_size=8, validation_data=(test_x, test_y))
    model_new.save('ddi.h5')
    # print(master_tokens)
    with open('ddi_vocab.json', 'w') as f:
        json.dump(master_tokens, f)
    # plt.plot(history.history['val_accuracy'])
    # plt.show()

    test_inp = np.asarray([train_x[0]])
    result = model_new.predict(test_inp)
    if np.argmax(result[0]) == 1:
        print("true")
    else:
        print("false")

    # text = 'MTX corticosteroids'
    # inp_x = infer(text, master_tokens)
    # inp_x = np.asarray([inp_x])
    # result = model_new.predict(inp_x)
    # # print(result)
    # if np.argmax(result[0]) == 1:
    #     print("true")
    # else:
    #     print("false")

def train_type(final_x, final_y_type_encoded, final_y_type, master_tokens, class_tokens):
    train_x, test_x, train_y, test_y = train_test_split(final_x, final_y_type_encoded, test_size=0.2)

    input_layer = Input(shape=(train_x.shape[1],))
    # model=Embedding(input_dim=len(master_data_vocab)+1,output_dim=32,input_length=train_x.shape[1])(input_layer)
    # model = Bidirectional(LSTM(units = 32,return_sequences=True, recurrent_dropout=0.2)) (model)
    model = Dense(32, activation='relu')(input_layer)
    output_layer = Dense(len(np.unique(final_y_type)), activation="softmax")(model)
    model_new = Model(input_layer, output_layer)

    model_new.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model_new.summary()

    history = model_new.fit(train_x, train_y, epochs=100, batch_size=16, validation_data=(test_x, test_y))
    model_new.save('type.h5')
    # print(master_tokens)
    with open('type_vocab.json', 'w') as f:
        json.dump(master_tokens, f)
    # plt.plot(history.history['val_accuracy'])
    # plt.show()

    test_inp = np.asarray([train_x[0]])
    result = model_new.predict(test_inp)
    for i in range(len(result)):
        print(class_tokens[np.argmax(result[0])])

    # text = 'MTX corticosteroids'
    # inp_x = infer(text, master_tokens)
    # inp_x = np.asarray([inp_x])
    # result = model_new.predict(inp_x)
    # # print(result)
    # for i in range(len(result)):
    #     print(class_tokens[np.argmax(result[0])])


path = '/Users/vsatpathy/Desktop/DDICorpus-master/APIforDDICorpus/DDICorpus/Train/DrugBank/'
filenames = os.listdir(path)

master_vocab = []
master_x = []
master_y_ddi = []
master_y_type = []
for file in filenames:
    filepath = path + file
    vocab, X, Y_ddi, Y_type = create_vocab(filepath)
    master_vocab.extend(vocab)
    master_x.extend(X)
    master_y_ddi.extend(Y_ddi)
    master_y_type.extend(Y_type)
master_vocab = sorted(set(master_vocab))
# print(len(master_vocab))

t = Tokenizer(num_words=len(master_vocab))
t.fit_on_texts(master_vocab)
master_tokens = t.word_index
# print(len(master_tokens))

# print(master_x[0], master_y_ddi[0], master_y_type[0])
final_x, master_data_vocab = create_dataset(master_x, master_tokens)
# print(final_x)

label_encoder = LabelEncoder()
final_y_ddi = label_encoder.fit_transform(master_y_ddi)
final_y_type = label_encoder.fit_transform(master_y_type)

class_tokens = {}
for i in range(len(final_y_type)):
    if final_y_type[i] not in class_tokens:
        class_tokens[final_y_type[i]] = master_y_type[i]
# print(class_tokens)
# y = final_y_ddi.reshape(final_y_ddi.shape[0],final_y_ddi.shape[1])
# y = to_categorical(y,num_classes=np.unique(y))

onehot_encoder = OneHotEncoder(sparse=False)
final_y_ddi = final_y_ddi.reshape(len(final_y_ddi), 1)
final_y_ddi = onehot_encoder.fit_transform(final_y_ddi)

final_y_type = final_y_type.reshape(len(final_y_type),1)
final_y_type_encoded = onehot_encoder.fit_transform(final_y_type)

train_ddi(final_x, final_y_ddi, master_tokens)
train_type(final_x, final_y_type_encoded, final_y_type, master_tokens, class_tokens)