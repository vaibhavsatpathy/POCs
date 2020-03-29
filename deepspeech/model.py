from keras.layers import Dense, Input, Embedding, LSTM, Bidirectional, Conv1D, Flatten, TimeDistributed
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras import losses
import tensorflow as tf

# model = Model()
# inp_layer = Input((1,16000))
# # x = Dense(128, activation='relu')(inp_layer)
# x = Conv1D(128,kernel_size=1,input_shape=(1,16000))(inp_layer)
# x = Flatten()(x)
# x = Embedding(input_dim=128,output_dim=64, input_shape=(16000,128))(x)
# x = Bidirectional(LSTM(64, return_sequences=True))(x)
# out_layer = Dense(26, activation='softmax')(x)
# model = Model(inp_layer,out_layer)
# # model.compile(optimizer='adam',loss='categorical_crossentropy')
# model.summary()


model = Model()
inp_layer = Input((1,16000))
# inp_layer = tf.placeholder(tf.float32,shape=final_sounds.shape)
x = TimeDistributed(Dense(128, activation='relu'))(inp_layer)
# x = Conv1D(max_len_trans,kernel_size=1,input_shape=(1,max_len_sound),trainable=False)(inp_layer)
# x = Flatten()(x)
# x = Embedding(input_dim=28,output_dim=64, input_shape=(1,128))(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
out_layer = Dense(27, activation='softmax')(x)

model = Model(inp_layer,out_layer)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()