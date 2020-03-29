import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import cv2

def model_train():
	batch_size=64
	epochs=2

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	y_train=to_categorical(y_train)
	y_test=to_categorical(y_test)
	X_train=X_train.reshape(X_train.shape[0],28,28,1)
	X_test=X_test.reshape(X_test.shape[0],28,28,1)

	x=Input(shape=(28,28,1))
	x1=Conv2D(32, kernel_size=(3,3), activation='relu',padding='same',strides=(2,2))(x)
	x1=Conv2D(64, kernel_size=(3,3), activation='relu')(x1)
	x1=BatchNormalization()(x1)
	x1=Dropout(0.2)(x1)
	x1=Conv2D(128, kernel_size=(3,3), activation='relu')(x1)
	x1=Conv2D(256, kernel_size=(3,3), activation='relu')(x1)
	x1=BatchNormalization()(x1)
	x1=Dropout(0.2)(x1)
	x1=Flatten()(x1)
	y=Dense(10,activation='softmax')(x1)
	model=Model(x,y)
	model.summary()

	opt=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	model.compile(metrics=['accuracy'],optimizer='adam',loss='categorical_crossentropy')
	model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=batch_size,epochs=epochs,verbose=1)
	model.save('simple_cnn.h5')

def rcnn(image,model,window_size,stride):
	coords=[]
	for i in range(0,image.shape[0],stride):
		for j in range(0,image.shape[1],stride):
			temp=image[i:i+window_size,j:j+window_size]
			if temp.shape==(window_size,window_size):
				temp=cv2.resize(temp,(28,28))
				temp=np.reshape(temp,(1,28,28,1))
				result=model.predict(temp)
				dummy=[]
				if (result[0][np.argmax(result)])>=0.9 and (np.argmax(result[0]))==1:
					dummy.append(i)
					dummy.append(j)
					coords.append(dummy)
	return coords

window_size=5
stride=5

model_train()

model=load_model('simple_cnn.h5')

image=cv2.imread('test.jpg',0)
image=cv2.bitwise_not(image)
image=cv2.resize(image,(28,28))
#image=np.reshape(image,(1,28,28,1))
# result=model.predict(image)
# print(np.argmax(result))
#(thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
coords=rcnn(image,model,window_size,stride)

print("all possible coords: ",coords)
# image2=cv2.circle(image,tuple(coords[0]),50,(0,0,255),5)
cv2.imshow('test',image)
cv2.waitKey(0)
cv2.destroyAllWindows()