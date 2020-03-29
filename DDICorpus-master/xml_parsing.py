import xml.etree.ElementTree as ET
import string
import os
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense, Embedding, Bidirectional, LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import json
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
	tree = ET.parse(file_path)
	root = tree.getroot()
	vocab = []
	Y = []
	X = []

	for child in root:
		sent = child.attrib['text']
		sentence = sent.split(" ")
		# print(sentence)
		sub_tag_dict ={}
		for sub_child in child:
			tag = sub_child.tag
			if tag == 'entity':
				tags = sub_child.attrib['type']
				tagged_word = sub_child.attrib['text']
				# print(len(tagged_word),tagged_word)
				if (len(tagged_word.split(' '))) > 1:
					for temp_word in tagged_word.split(' '):
						sub_tag_dict[temp_word] = tags
				else:
					sub_tag_dict[tagged_word] = tags
				# print(tags,"---->",tagged_word)
		# print(sub_tag_dict)
		for word in sentence:
			word = pre_process(word)
			X.append(word)
			if word in sub_tag_dict:
				Y.append(sub_tag_dict[word])
			else:
				Y.append('other')
			vocab.append(word)
	return vocab, X, Y


def create_dataset(X, Y, tokens, ent_dict):
	final_x = []
	final_y = []

	for i in range(len(X)):
		temp_x = []
		temp_y = []
		for j in range(len(X[i])):
			if X[i][j] in tokens:
				temp_x.append(tokens[X[i][j]])
				temp_y.append(ent_dict[Y[i][j]])
			else:
				pass
		temp_x = np.asarray(temp_x)
		temp_y = np.asarray(temp_y)
		final_x.append(temp_x)
		final_y.append(temp_y)
	return np.asarray(final_x), np.asarray(final_y)


path = '/Users/vsatpathy/Desktop/DDICorpus-master/APIforDDICorpus/DDICorpus/Train/DrugBank/'
filenames = os.listdir(path)

master_vocab = []
master_x = []
master_y = []
for file in filenames:
	file_path = path + file
	bow, X, Y = create_vocab(file_path)
	master_vocab.extend(bow)
	master_x.append(X)
	master_y.append(Y)

t=Tokenizer(num_words=len(master_vocab))
t.fit_on_texts(master_vocab)
master_tokens = t.word_index
# master_vocab = set(master_vocab)
reverse_master_tokens = {}
for key,val in master_tokens.items():
	reverse_master_tokens[val] = key
# print(reverse_master_tokens)

entities = []
for label in master_y:
	tags = np.unique(label)
	for tag in tags:
		if tag not in entities:
			entities.append(tag)
entities = sorted(entities)

master_entities = {}
for ent in entities:
	master_entities[ent] = len(master_entities)

final_x, final_y = create_dataset(master_x, master_y, master_tokens, master_entities)
# final_x = np.asarray(final_x)
# final_y = np.asarray(final_y)

max_length = 20
X=pad_sequences(final_x,maxlen=max_length,padding='post',dtype=float)
y=pad_sequences(final_y,maxlen=max_length,padding='post')
y = y.reshape(y.shape[0],y.shape[1])
y = to_categorical(y,num_classes=len(master_entities))
# print(y.shape)

train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2)

input_layer = Input(shape=(train_x.shape[1],))
model = Embedding(input_dim=len(master_vocab) + 1, output_dim=32, input_length=train_x.shape[1])(input_layer)
model = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.2))(model)
output_layer = Dense(len(master_entities), activation="softmax")(model)
model_new = Model(input_layer, output_layer)

model_new.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model_new.summary()

history = model_new.fit(train_x, train_y, epochs=20, batch_size=8, validation_data=(test_x, test_y))
model_new.save('entity.h5')
# print(master_tokens)
with open('entity_vocab.json','w') as f:
	json.dump(master_tokens,f)
# plt.plot(history.history['val_accuracy'])
# plt.show()

reverse_master_entities = {}
for key, val in master_entities.items():
	reverse_master_entities[val] = key
# with open('entities.json','w') as f:
# 	json.dump(reverse_master_entities,f)

# print(train_x[0])
test_input = np.asarray([train_x[0]])
result = model_new.predict(test_input)
# print(result.shape)
for i in range(len(test_input[0])):
	# print(np.argmax(result[0][i]))
	if test_input[0][i] != 0.0:
		print(reverse_master_entities[np.argmax(result[0][i])], "---->", reverse_master_tokens[test_input[0][i]])
	else:
		pass