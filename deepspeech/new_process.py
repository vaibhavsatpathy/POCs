import soundfile as sf
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import progressbar
import librosa
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt

path = '/Users/vsatpathy/Desktop/deepspeech/LibriSpeech 2/train-clean-100/'
scaler = MinMaxScaler()
std = StandardScaler(with_mean=True,with_std=True)

path_vocab = '/Users/vsatpathy/Desktop/deepspeech/words.txt'
data = open(path_vocab)
lines = data.readlines()
vocab = {}
for line in lines:
    words = line.split(' ')
    words[-1] = words[-1].split('\n')[0]
    key = words[0]
    words = ''.join(words[1:])
    vocab[key] = words

reverse_vocab = {}
for key,val in vocab.items():
    reverse_vocab[int(val)] = key
reverse_vocab[0] = ' '
# print(reverse_vocab)


def process_sent(sent, vocab):
    len_trans = len(sent)
    tok_sent = []
    for char in sent:
        if char not in vocab:
            tok_sent.append('0')
        else:
            tok_sent.append(vocab[char])
    return tok_sent, len_trans


all_sound_files = []
all_trans_files = []
sub_dir_1 = os.listdir(path)
for i in range(len(sub_dir_1)):
    sub_path_1 = path + sub_dir_1[i] + '/'
    for sub_dir_2 in os.listdir(sub_path_1):
        sub_path_2 = sub_path_1 + sub_dir_2 + '/'
        for sub_dir_3 in os.listdir(sub_path_2):
            if (sub_dir_3.split('.')[-1]) != 'txt':
                final_path = sub_path_2 + sub_dir_3
                all_sound_files.append(final_path)
            else:
                all_trans_files.append(sub_path_2 + sub_dir_3)
all_sound_files = sorted(all_sound_files)
all_trans_files = sorted(all_trans_files)
# print(len(all_sound_files))

transcript = []
for i in progressbar.progressbar(range(len(all_trans_files))):
    f = open(all_trans_files[i])
    lines = f.readlines()
    for line in lines:
        words = line.split(' ')
        words[-1] = words[len(words) - 1].split('\n')[0]
        sent = ' '.join(words[1:]).lower()
        transcript.append(sent)
# print(len(transcript))

# max_len = 100000
# data, samplerate = sf.read('/Users/vsatpathy/Desktop/deepspeech/LibriSpeech/test-clean/61/70968/61-70968-0000.flac')
# print(data.shape, samplerate)
# data_new = np.pad(data,max_len-data.shape[0],mode='mean')
# data_new = data_new[max_len-data.shape[0]:]
# data_new = scaler.fit_transform(data_new.reshape(-1,1))
# print(data_new.shape, samplerate)
# plt.plot(data_new)
# plt.show()

master_sound = []
max_len_sound = 0
max_len_trans = 0
summation = 0
for i in progressbar.progressbar(range(len(all_sound_files))):
    data, samplerate = sf.read(all_sound_files[i])
    if data.shape[0] > max_len_sound:
        max_len_sound = data.shape[0]
    curr_trans = transcript[i]
    if len(curr_trans) > max_len_trans:
        max_len_trans = len(curr_trans)
    n = data.shape[0]
    m = len(curr_trans)
    summation += float(n/m)
    master_sound.append(data)

max_feat_per_window = int(summation/len(all_sound_files))
# print(max_feat_per_window, max_len_sound, max_len_trans)

master_data = np.zeros(shape=(len(all_sound_files),max_len_trans,max_feat_per_window))
for j in progressbar.progressbar(range(len(master_sound))):
    data = master_sound[j]
    data_new = np.pad(data,max_len_sound-data.shape[0],mode='mean')
    data_new = data_new[max_len_sound-data.shape[0]:]
    # data_new = scaler.fit_transform(data_new.reshape(-1,1))
    curr_voice = np.zeros((max_len_trans,max_feat_per_window))
    counter = 0
    for i in range(0,len(data_new),max_feat_per_window):
        temp_data = data[i:i+max_feat_per_window]
        # print(temp_data.shape)
        if len(temp_data) < max_feat_per_window:
            temp_data = np.pad(temp_data,max_feat_per_window-temp_data.shape[0],'constant',constant_values=0)
            temp_data = temp_data[abs(max_feat_per_window-temp_data.shape[0]):]
            curr_voice[counter] = temp_data
        else:
            curr_voice[counter] = temp_data
    master_data[j] = curr_voice

# print(master_data.shape)
master_trans = []
for i in range(len(transcript)):
    tokenized_sent,_ = process_sent(transcript[i], vocab)
    master_trans.append(tokenized_sent)
master_trans = pad_sequences(master_trans,maxlen=max_len_trans,padding='post')
master_trans = to_categorical(master_trans,num_classes=len(vocab)+1)
# print(master_trans.shape)


def inference(model, master_data, reverse_vocab):
    inp_test = np.asarray([master_data[0]])
    result = model.predict(inp_test)
    print(result)
    chars = []
    for i in range(result.shape[1]):
        char = reverse_vocab[np.argmax(result[0][i])]
        chars.append(char)
    print(' '.join(chars))


from keras.layers import Dense, Input, Embedding, LSTM, Bidirectional, Conv1D, Flatten, TimeDistributed, Dropout, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint

model = Model()
inp_layer = Input(shape=(master_data.shape[1],master_data.shape[2]))
x = TimeDistributed(Dense(1024,activation='relu'))(inp_layer)
x = TimeDistributed(Dropout(0.2))(x)
x = TimeDistributed(Dense(1024,activation='relu'))(x)
x = TimeDistributed(Dropout(0.2))(x)
x = TimeDistributed(Dense(512,activation='relu'))(x)
x = TimeDistributed(Dropout(0.2))(x)
# x = Conv1D(max_len_trans,kernel_size=1,input_shape=(1,max_len_sound),trainable=False)(inp_layer)
# x = Flatten()(x)
# x = Embedding(input_dim=len(vocab)+2,output_dim=64, input_length=max_len_trans)(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
# x = Flatten()(x)
out_layer = Dense(len(vocab)+1, activation='softmax')(x)
# out_layer = TimeDistributed(Dense(max_len_trans, activation='softmax'))(x)

model = Model(inp_layer,out_layer)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint('model.h5')
callbacks = [checkpoint]
model.fit(master_data,master_trans,batch_size=32,epochs=100, callbacks=callbacks)

inference(model, master_data, reverse_vocab)