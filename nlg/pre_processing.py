from __future__ import print_function
from keras.preprocessing.text import text_to_word_sequence
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import csv
import re

data_path = 'e2e-dataset/trainset.csv'

def infer(input_keys, input_token_index):
    input_texts = []

    name_text = input_keys['name']
    eatType_text = input_keys['eatType']
    food_text = input_keys['food']
    priceRange_text = input_keys['priceRange']
    customerRating_text = input_keys['customerRating']
    area_text = input_keys['area']
    kidsFriendly_text = input_keys['familyFriendly']
    near_text = input_keys['near']

    name_string = 'start_name ' + name_text + ' stop_name' if name_text else 'start_name stop_name'
    eatType_string = 'start_eattype ' + eatType_text + ' stop_eattype' if eatType_text else 'start_eattype stop_eattype'
    food_string = 'start_food ' + food_text + ' stop_food' if food_text else 'start_food stop_food'
    priceRange_string = 'start_pricerange ' + priceRange_text + ' stop_pricerange' if priceRange_text else 'start_pricerange stop_pricerange'
    customerRating_string = 'start_customerrating ' + customerRating_text + ' stop_customerrating' if customerRating_text else 'start_customerrating stop_customerrating'
    area_string = 'start_area ' + area_text + ' stop_area' if area_text else 'start_area stop_area'
    kidsFriendly_string = 'start_kidsfriendly ' + kidsFriendly_text + ' stop_kidsfriendly' if kidsFriendly_text else 'start_kidsfriendly stop_kidsfriendly'
    near_string = 'start_near ' + near_text + ' stop_near' if near_text else 'start_near stop_near'

    input_string = ' '.join(
        [name_string, eatType_string, food_string, priceRange_string, customerRating_string, area_string,
         kidsFriendly_string, near_string])
    input_texts.append(input_string)
    print(input_texts[0])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')

    for i, (input_text) in enumerate(input_texts):
        for t, wor in enumerate(input_text.split(" ")):
            encoder_input_data[i, t, input_token_index[wor]] = 1.

    return encoder_input_data


def decode_sequence(input_seq, enc_model, dec_model):
    # print(input_seq)
    states_value = enc_model.predict(input_seq)  # Encode the input
    target_seq = np.zeros((1, 1, num_decoder_tokens))  # Empty target sequence of length 1
    target_seq[0, 0, target_token_index['\t']] = 1.  # Start token for target

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = dec_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_wor = reverse_target_token_index[sampled_token_index]
        decoded_sentence += ' ' + sampled_wor

        # Exit condition: either hit max length or find stop token
        if (sampled_wor == '\n' or
                len(decoded_sentence) > max_decoder_seq_length * 2):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

input_texts = []
target_texts = []
input_vocab = set()
target_vocab = set()

with open(data_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    training_set = list(reader)

for element in training_set[1:]:
    input_text = element[0]

    name_text = re.search('(?<=name\[).+?(?=\])', input_text)
    eatType_text = re.search('(?<=eatType\[).+?(?=\])', input_text)
    food_text = re.search('(?<=food\[).+?(?=\])', input_text)
    priceRange_text = re.search('(?<=priceRange\[).+?(?=\])', input_text)
    customerRating_text = re.search('(?<=customer rating\[).+?(?=\])', input_text)
    area_text = re.search('(?<=area\[).+?(?=\])', input_text)
    kidsFriendly_text = re.search('(?<=familyFriendly\[).+?(?=\])', input_text)
    near_text = re.search('(?<=near\[).+?(?=\])', input_text)

    name_string = 'start_name ' + name_text.group(0) + ' stop_name' if name_text else 'start_name stop_name'
    eatType_string = 'start_eatType ' + eatType_text.group(
        0) + ' stop_eatType' if eatType_text else 'start_eatType stop_eatType'
    food_string = 'start_food ' + food_text.group(0) + ' stop_food' if food_text else 'start_food stop_food'
    priceRange_string = 'start_priceRange ' + priceRange_text.group(
        0) + ' stop_priceRange' if priceRange_text else 'start_priceRange stop_priceRange'
    customerRating_string = 'start_customerRating ' + customerRating_text.group(
        0) + ' stop_customerRating' if customerRating_text else 'start_customerRating stop_customerRating'
    area_string = 'start_area ' + area_text.group(0) + ' stop_area' if area_text else 'start_area stop_area'
    kidsFriendly_string = 'start_kidsFriendly ' + kidsFriendly_text.group(
        0) + ' stop_kidsFriendly' if kidsFriendly_text else 'start_kidsFriendly stop_kidsFriendly'
    near_string = 'start_near ' + near_text.group(0) + ' stop_near' if near_text else 'start_near stop_near'

    input_string = ' '.join(
        [name_string, eatType_string, food_string, priceRange_string, customerRating_string, area_string,
         kidsFriendly_string, near_string])
    input_texts.append(input_string)

    target_text = element[1]
    target_text = '\t ' + target_text + ' \n'
    target_texts.append(target_text)
    # print(input_string)
    # print(target_text)

input_vocab = set(text_to_word_sequence(" ".join(input_texts), filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~'))
target_vocab = set(text_to_word_sequence(" ".join(target_texts), filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~'))

input_text_modif = []
for input_text in input_texts:
    input_text_modif.append(' '.join(text_to_word_sequence(input_text, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~', lower=True)))

target_text_modif = []
for target_text in target_texts:
    target_text_modif.append(' '.join(text_to_word_sequence(target_text, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~', lower=True)))

input_texts = input_text_modif
target_texts = target_text_modif

input_vocab = sorted(list(input_vocab))
target_vocab = sorted(list(target_vocab))
num_encoder_tokens = len(input_vocab)
num_decoder_tokens = len(target_vocab)
max_encoder_seq_length = max([len(txt.split(" ")) for txt in input_texts])
max_decoder_seq_length = max([len(txt.split(" ")) for txt in target_texts])
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(wor, i) for i, wor in enumerate(input_vocab)])
target_token_index = dict(
    [(wor, i) for i, wor in enumerate(target_vocab)])
# print(input_token_index)

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, wor in enumerate(input_text.split(" ")):
        encoder_input_data[i, t, input_token_index[wor]] = 1.
    for t, wor in enumerate(target_text.split(" ")):
        decoder_input_data[i, t, target_token_index[wor]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[wor]] = 1.


batch_size = 64    # Batch size for training.
epochs = 15     # Number of epochs to train for.
latent_dim = 256   # Latent dimensionality of the encoding space.
# Build encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))      # unique input tokens
encoder = LSTM(latent_dim, return_state=True)                 # number of neurons
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]                           # We discard the outputs
# Build the decoder, using the context of the encoder

decoder_inputs = Input(shape=(None, num_decoder_tokens))                  # unique output tokens
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) # internal states for inference on new data
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.summary()

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('master_model.h5')

# Inference model architecture
encoder_model = Model(encoder_inputs, encoder_states)
# encoder_model.summary()
encoder_model.save('encoder_model.h5')

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
# decoder_model.summary()
decoder_model.save('decoder_model.h5')

reverse_input_token_index = dict(
    (i, wor) for wor, i in input_token_index.items())
reverse_target_token_index = dict(
    (i, wor) for wor, i in target_token_index.items())

enc_mod = load_model('encoder_model.h5')
dec_mod = load_model('decoder_model.h5')

graph()

for seq_index in range(1):
    # Take one sequence (part of the training set)
    # for trying out decoding.

    input_keys = {'name': 'the mill', 'eatType': 'pub', 'food': '', 'priceRange': 'high', 'customerRating': '5',
                  'area': 'city centre', 'familyFriendly': 'no', 'near': ''}
    input_seq_user = infer(input_keys, input_token_index)
    decoded_sentence = decode_sequence(input_seq_user, enc_mod, dec_mod)

    # input_seq = encoder_input_data[seq_index: seq_index + 1]
    # decoded_sentence = decode_sequence(input_seq, enc_mod, dec_mod)

    print('-')
    # print('Input sentence:\n', input_texts[seq_index])
    print('Decoded sentence:\n', decoded_sentence)
    print('-')