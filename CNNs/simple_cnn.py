import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD
import numpy as np

batch_size=64
epochs=2

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(y_train[0])
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