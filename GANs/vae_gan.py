import numpy

from keras.layers import Input,multiply,Embedding
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)
y_train=y_train.reshape(-1,1)

def sampling(args):
	mean,stddev=args
	epsilon=K.random_normal(shape=(K.shape(mean)[0], 512), mean=0., stddev=1.0)
	sample=mean+K.exp(stddev/2)*epsilon
	return sample

def encoder(filter,size,rows,cols,channels):
	inp=Input(shape=(rows*cols*channels))

	enc=Conv2D(filters=filter,kernel_size=size,strides=2,padding='same')(inp)
	enc=BatchNormalization(epsilon=1e-5)(enc)
	enc=LeakyReLU(alpha=0.2)(enc)

	enc=Conv2D(filters=filter*2,kernel_size=size,strides=2,padding='same')(inp)
	enc=BatchNormalization(epsilon=1e-5)(enc)
	enc=LeakyReLU(alpha=0.2)(enc)

	enc=Conv2D(filters=filter*4,kernel_size=size,strides=2,padding='same')(inp)
	enc=BatchNormalization(epsilon=1e-5)(enc)
	enc=LeakyReLU(alpha=0.2)(enc)

	enc=Conv2D(filters=filter*8,kernel_size=size,strides=2,padding='same')(inp)
	enc=BatchNormalization(epsilon=1e-5)(enc)
	enc=LeakyReLU(alpha=0.2)(enc)

	enc=Flatten()(enc)
	mean=Dense(512)(enc)
	stddev=Dense(512,activation='tanh')(enc)

	latent=Lambda(sampling,output_shape=(512))([mean,stddev])
	encoder=Model([inp],[mean,stddev,latent])
	encoder.compile(loss='mse',optimizer=SGD(lr=0.0003))
	return encoder

def decoder(filter,size,rows,cols,channels):
	inp=Input(shape=(512,))

	dec=Dense(filter*8*rows*cols*channels)(inp)
	dec=Reshape((rows,cols,filter*8))(dec)
	dec=BatchNormalization(epsilon=1e-5)(dec)
	dec=Activation('relu')(dec)

	dec=Conv2DTranspose(filters=filter*4,strides=2,padding='same',kernel_size=size)(dec)
	dec=BatchNormalization(epsilon=1e-5)(dec)
	dec=Activation('relu')(dec)

	dec=Conv2DTranspose(filters=filter*2,strides=2,padding='same',kernel_size=size)(dec)
	dec=BatchNormalization(epsilon=1e-5)(dec)
	dec=Activation('relu')(dec)

	dec=Conv2DTranspose(filters=filter,strides=2,padding='same',kernel_size=size)(dec)
	dec=BatchNormalization(epsilon=1e-5)(dec)
	dec=Activation('relu')(dec)

	dec=Conv2DTranspose(filters=rows*cols*channels,strides=2,padding='same',kernel_size=size)(dec)
	dec=Activation('tanh')(dec)

	decoder=Model([inp],[dec])
	decoder.compile(loss='mse',optimizer=SGD(lr=0.0003))
	return decoder

def discriminator(filter,size,rows,cols,channels):
	inp=Input(shape=(rows*cols*channels))

	disc=Conv2D(filters=filter,kernel_size=size,strides=2,padding='same')(inp)
	disc=LeakyReLU(alpha=0.2)(enc)

	disc=Conv2D(filters=filter*2,kernel_size=size,strides=2,padding='same')(inp)
	disc=BatchNormalization(epsilon=1e-5)(enc)
	disc=LeakyReLU(alpha=0.2)(enc)

	disc=Conv2D(filters=filter*4,kernel_size=size,strides=2,padding='same')(inp)
	disc=BatchNormalization(epsilon=1e-5)(enc)
	disc=LeakyReLU(alpha=0.2)(enc)

	disc=Conv2D(filters=filter*8,kernel_size=size,strides=2,padding='same')(inp)
	disc=BatchNormalization(epsilon=1e-5)(enc)
	disc=LeakyReLU(alpha=0.2)(enc)

	disc=Conv2D(filters=filter*8,kernel_size=size,strides=2,padding='same')(inp)
	discr=BatchNormalization(epsilon=1e-5)(disc)
	discr=LeakyReLU(alpha=0.2)(disc)
	discr=Flatten()(disc)
	discr=Dense(1,activation='sigmoid')(disc)

	discriminator=Model([inp],[disc,discr])
	discriminator.compile(loss='mse',optimizer=SGD(lr=0.0003))
	return discriminator

def VAE(rows,cols,channels,model_encoder,model_decoder):
	inp=Input(shape=(rows,cols,channels))
	mean,stddev,latent=model_encoder(inp)
	output=model_decoder(latent)
	VAE=Model([inp],[output])

	kl = - 0.5 * K.sum(1 + stddev - K.square(mean) - K.exp(stddev), axis=-1)
	crossent = 64 * metrics.mse(K.flatten(inp), K.flatten(output))
	VAEloss = K.mean(crossent + kl)
	VAE.add_loss(VAEloss)
	VAE.compile(optimizer=SGD(lr=0.0003))

	return VAE

def train(batch_size,max_iterations,model_encoder,model_decoder,model_discriminator,model_VAE):
	input_layer=Input(shape=(rows,cols,channels))
	mean,stddev,lat=model_encoder(input_layer)
	out=model_decoder(lat)
	out_gen=model_decoder(mean+stddev)

	D_true,Di_true=model_discriminator(input_layer)
	D_fake_gen,Di_fake_gen=model_discriminator(out)
	D_gen,Di_gen=model_discriminator(out_gen)

	valid=np.ones((batch_size,1))
	fake=np.zeros((batch_size,1))
	
	for epoch in range(0,max_iterations):
		noise=np.random.normal(0, 1, (batch_size, 256))
		index=np.random.randint(0, X_train.shape[0], size=batch_size)
		image_batch = X_train[index]

		latent_vec=model_encoder.predict(X_train)
		real_enc_image=model_decoder.predict(latent_vec)
		fake_enc_image=model_decoder.predict(noise)

		gen_dec_loss_enc=model_decoder.train_on_batch(latent_vector,valid)
		gen_dec_loss_fake=model_decoder.train_on_batch(noise,valid)
		vae_loss=model_VAE.train_on_batch(X_train,None)

		disc_loss_fake=D_fake_gen.train_on_batch(fake_enc_image,fake)
		disc_loss_enc=D_gen.train_on_batch(real_enc_image,valid)
		disc_loss_true=D_true.train_on_batch(X_train,valid)

batch_size=32
rows=28
cols=28
channels=1
max_iterations=200000
model_encoder=encoder(32,5,rows,cols,channels)
model_decoder=encoder(32,5,rows,cols,channels)
model_discriminator=encoder(32,5,rows,cols,channels)
model_VAE=VAE(rows,cols,channels,model_encoder,model_decoder)

train(batch_size,max_iterations,model_encoder,model_decoder,model_discriminator,model_VAE)
