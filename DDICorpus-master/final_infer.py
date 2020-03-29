import xml.etree.ElementTree as ET
import string
import os
import json
from keras.models import load_model
import numpy as np

path = '/Users/vsatpathy/Desktop/DDICorpus-master/APIforDDICorpus/DDICorpus/Train/DrugBank/'
filenames = os.listdir(path)

path_to_json = '/Users/vsatpathy/Desktop/DDICorpus-master/'


def remove_punc(text):
	no_punc = "".join([c for c in text if c not in string.punctuation])
	return no_punc


def pre_process(word):
	final_word = word.lower()
	final_word = final_word.strip()
	final_word = remove_punc(final_word)
	return final_word


def create_inp(filepath):
	tree = ET.parse(filepath)
	root = tree.getroot()
	X = []

	for child in root:
		sent = child.attrib['text']
		sentence = sent.split(" ")
		for word in sentence:
			word = pre_process(word)
			X.append(word)
	return X


def pad_inp(inp_x, max_length):
	final_x = []
	# print(inp_x)
	for i in range(max_length):
		if i >= len(inp_x):
			final_x.append(0.0)
		else:
			final_x.append(float(inp_x[i]))
	# print(final_x)
	return np.asarray(final_x)


master_x = []
for file in filenames:
	file_path = path + file
	sample_x = create_inp(file_path)
	master_x.append(sample_x)

model_entity = load_model('entity.h5')
model_ddi = load_model('ddi.h5')
model_type = load_model('type.h5')

with open(path_to_json + 'entity_vocab.json') as vocab:
	data = json.load(vocab)

	dummy_inp = []
	for word in master_x[0]:
		dummy_inp.append(data[word])

	dummy_inp = pad_inp(dummy_inp, 20)
	dummy_inp = np.asarray([dummy_inp])
	result = model_entity.predict(dummy_inp)