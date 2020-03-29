import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input,multiply,Embedding
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)
y_train=y_train.reshape(-1,1)

def generator():
	gen=Sequential()
	gen.add(Dense(256,input_dim=100))
	gen.add(LeakyReLU(0.2))
	gen.add(BatchNormalization(momentum=0.8))
	gen.add(Dense(512))
	gen.add(LeakyReLU(0.2))
	gen.add(BatchNormalization(momentum=0.8))
	gen.add(Dense(1024))
	gen.add(LeakyReLU(0.2))
	gen.add(BatchNormalization(momentum=0.8))
	gen.add(Dense(784,activation='tanh'))
	gen.summary()

	noise=Input(shape=(100,))
	label=Input(shape=(1,),dtype='int32')
	label_embedding=Flatten()(Embedding(10,100)(label))
	model_input=multiply([noise,label_embedding])
	image=gen(model_input)

	gen=Model([noise,label],image)
	gen.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002, beta_1=0.5))
	return gen


def discriminator():
	disc=Sequential()
	disc.add(Dense(512,input_dim=784))
	disc.add(LeakyReLU(0.2))
	disc.add(Dropout(0.4))
	disc.add(Dense(512))
	disc.add(LeakyReLU(0.2))
	disc.add(Dropout(0.4))
	disc.add(Dense(512))
	disc.add(LeakyReLU(0.2))
	disc.add(Dropout(0.4))
	disc.add(Dense(1,activation='sigmoid'))
	disc.summary()

	image=Input(shape=(784,))
	label=Input(shape=(1,),dtype='int32')
	label_embedding=Flatten()(Embedding(10,784)(label))
	model_input=multiply([image,label_embedding])
	prediction=disc(model_input)

	disc=Model([image,label],prediction)
	disc.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002, beta_1=0.5),metrics=['accuracy'])
	return disc


def stacked_GAN(gen,disc):
	gan_input=Input(shape=(100,))
	label=Input(shape=(1,))
	x=gen([gan_input,label])
	disc.trainable=False
	gan_out=disc([x,label])
	gan_stack=Model([gan_input,label],gan_out)
	gan_stack.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002, beta_1=0.5))
	return gan_stack

def test(gen,i):
	noise=np.random.normal(0,1,(1,100))
	label=np.random.randint(0,10,1).reshape(-1,1)
	image=np.squeeze(gen.predict([noise,label]),axis=0)
	plt.imsave('/home/vaibhav/deep_learning/gan/code/images2/epoch_%d_tag_%s'%(i,label[0]),image.reshape(28,28),format='jpg',cmap='gray')

def train(max_iter,batch_size,gen,disc,gan_stack):
	valid=np.ones((batch_size,1))
	fake=np.zeros((batch_size,1))
	for i in range(max_iter):
		noise=np.random.normal(0,1,(batch_size,100))
		index=np.random.randint(0, X_train.shape[0], size=batch_size)
		image_batch = X_train[index]
		label_batch = y_train[index]

		fake_images=gen.predict([noise,label_batch])

		disc.trainable=True
		disc_loss_real=disc.train_on_batch([image_batch,label_batch],valid)
		disc_loss_fake=disc.train_on_batch([fake_images,label_batch],fake)
		disc_loss_final=0.5*np.add(disc_loss_real,disc_loss_fake)

		fake_labels=np.random.randint(0,10,batch_size).reshape(-1,1)
		disc.trainable=False
		gen_loss=gan_stack.train_on_batch([noise,fake_labels],valid)

		print('epoch_%d---->gen_loss:[%f]---->disc_loss:[%f]---->acc:[%f]'%(i,gen_loss,disc_loss_final[0],disc_loss_final[1]*100))
		if i%100==0:
			#gen.save_weights('/home/vaibhav/deep_learning/gan/code/gen_weights/epoch_%d.h5'%i)
			#disc.save_weights('/home/vaibhav/deep_learning/gan/code/disc_weights/epoch_%d.h5'%i)
			test(gen,i)
			pass

gen=generator()
disc=discriminator()
gan_stack=stacked_GAN(gen,disc)
batch_size=32
max_iter=20000
train(max_iter,batch_size,gen,disc,gan_stack)
