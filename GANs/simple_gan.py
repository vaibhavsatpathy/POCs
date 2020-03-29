import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input
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

def generator():
	gen=Sequential()
	gen.add(Dense(256,input_dim=100))
	gen.add(LeakyReLU(0.2))
	gen.add(Dense(512))
	gen.add(LeakyReLU(0.2))
	gen.add(Dense(1024))
	gen.add(LeakyReLU(0.2))
	gen.add(Dense(784,activation='tanh'))
	gen.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002, beta_1=0.5))
	return gen


def discriminator():
	disc=Sequential()
	disc.add(Dense(1024,input_dim=784))
	disc.add(LeakyReLU(0.2))
	disc.add(Dropout(0.2))
	disc.add(Dense(512))
	disc.add(LeakyReLU(0.2))
	disc.add(Dropout(0.2))
	disc.add(Dense(256))
	disc.add(LeakyReLU(0.2))
	disc.add(Dropout(0.2))
	disc.add(Dense(1,activation='sigmoid'))
	disc.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002, beta_1=0.5))
	return disc


def stacked_GAN(gen,disc):
	disc.trainable=False
	gan_input=Input(shape=(100,))
	x=gen(gan_input)
	gan_out=disc(x)
	gan_stack=Model(inputs=gan_input,outputs=gan_out)
	gan_stack.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002, beta_1=0.5))
	return gan_stack


def test(gen,i):
	noise=np.random.normal(0,1,(1,100))
	image=np.squeeze(gen.predict(noise),axis=0)
	plt.imsave('/home/vaibhav/deep_learning/gan/code/images2/epoch_%d'%i,image.reshape(28,28),format='jpg',cmap='gray')

def train(max_iter,batch_size,gen,disc,gan_stack):
	for i in range(0,max_iter):
		noise=np.random.normal(0,1,(batch_size,100))
		image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

		fake_images=gen.predict(noise)

		final_images=np.concatenate([image_batch,fake_images])
		final_labels=np.concatenate((np.ones((np.int64(batch_size), 1)), np.zeros((np.int64(batch_size), 1))))

		disc.trainable=True
		disc_loss=disc.train_on_batch(final_images,final_labels)

		disc.trainable=False
		noise_gen=np.random.normal(0,1,(batch_size,100))
		y_mis_labels=np.ones(batch_size)
		gen_loss=gan_stack.train_on_batch(noise,y_mis_labels)

		print('epoch_%d---->gen_loss:[%f]---->disc_loss:[%f]'%(i,gen_loss,disc_loss))
		if i%100==0:
			#gen.save_weights('/home/vaibhav/deep_learning/gan/code/gen_weights/epoch_%d.h5'%i)
			#disc.save_weights('/home/vaibhav/deep_learning/gan/code/disc_weights/epoch_%d.h5'%i)
			test(gen,i)
			pass

gen=generator()
disc=discriminator()
gan_stack=stacked_GAN(gen,disc)


max_iter=20000
batch_size=32
train(max_iter,batch_size,gen,disc,gan_stack)
