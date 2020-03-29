import nnabla as nn
import nnabla.functions as func
import nnabla.parametric_functions as pfunc
import nnabla.solvers as sol

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def generator(x):
	with nn.parameter_scope("generator"):
		with nn.parameter_scope("block1"):
			x=pfunc.affine(x,256)
			x=func.leaky_relu(x,0.2)
		with nn.parameter_scope("block2"):
			x=pfunc.affine(x,512)
			x=func.leaky_relu(x,0.2)
		with nn.parameter_scope("block3"):
			x=pfunc.affine(x,1024)
			x=func.leaky_relu(x,0.2)
		with nn.parameter_scope("block4"):
			x=pfunc.affine(x,784)
			x=func.tanh(x)
	return x

def discriminator(x):
	with nn.parameter_scope("discriminator"):
		with nn.parameter_scope("block1"):
			x=pfunc.affine(x,1024)
			x=func.leaky_relu(x,0.2)
		with nn.parameter_scope("block2"):
			x=pfunc.affine(x,512)
			x=func.leaky_relu(x,0.2)
		with nn.parameter_scope("block3"):
			x=pfunc.affine(x,256)
			x=func.leaky_relu(x,0.2)
		with nn.parameter_scope("block4"):
			x=pfunc.affine(x,1)
			x=func.sigmoid(x)
	return x	

def train(batch_size,X_train,max_iter):
	from nnabla.ext_utils import get_extension_context
	context="cpu"
	ctx=get_extension_context(context,device_id="0",type_config="float")
	nn.set_default_context(ctx)

	z=nn.Variable([batch_size,100,1,1])
	fake=generator(z)
	fake.persistent=True
	pred_fake=discriminator(fake)
	labels=func.constant(1,pred_fake.shape)
	loss_gen=func.mean(func.sigmoid_cross_entropy(pred_fake,labels))

	fake_disc=fake.get_unlinked_variable(need_grad=True)
	pred_fake_disc=discriminator(fake_disc)
	disc_fake_label=func.constant(0,pred_fake_disc.shape)
	loss_disc_fake=func.mean(func.sigmoid_cross_entropy(pred_fake_disc,disc_fake_label))

	r=nn.Variable([batch_size,784])
	real_pred=discriminator(r)
	disc_real_label=func.constant(0,real_pred.shape)
	loss_disc_real=func.mean(func.sigmoid_cross_entropy(pred_fake_disc,disc_real_label))

	loss_disc=loss_disc_real+loss_disc_fake

	solver_gen=sol.Adam(0.0002,beta1=0.5)
	solver_disc=sol.Adam(0.0002,beta1=0.5)

	for i in range(0,max_iter):
		index=np.random.randint(0, X_train.shape[0], size=batch_size)
		input_image=X_train[index]

		r.d=input_image
		z.d=np.random.randn(*z.shape)

		solver_gen.zero_grad()
		loss_gen.forward(clear_no_need_grad=True)
		loss_gen.backward(clear_buffer=True)
		solver_gen.weight_decay(0.0001)
		solver_gen.update()

		solver_disc.zero_grad()
		loss_disc.forward(clear_no_need_grad=True)
		loss_disc.backward(clear_buffer=True)
		solver_disc.weight_decay(0.0001)
		solver_disc.update()

		print("epoch-->[%d]-------loss_generator-->[%f]-------loss_discriminator-->[%f]"%(i,loss_gen.d,loss_disc.d))

		if i%100==0:
			with nn.parameter_scope("generator"):
				nn.save_parameters("/home/vaibhav/deep_learning/gan/code/gen_weights/epoch_%d.h5"%i)
			with nn.parameter_scope("discriminator"):
				nn.save_parameters("/home/vaibhav/deep_learning/gan/code/disc_weights/epoch_%d.h5"%i)


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)

batch_size=32
max_iter=20000
train(batch_size,X_train,max_iter)
