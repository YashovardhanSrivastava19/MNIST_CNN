""" 
Convolutional Neural Network --> a deep,feed-forward ANN in which the neural network preserves the hierarchical structure 
by learning internal feature representations and generalizing the features. 

Layers in CNN: 

-->  Input 
-->  Convolution(contains filters & image maps) 
-->  Subsampling(average pooling w/ learnable weights per feature map) 
-->  Output

Rotational,Trainslation and Scale invariance can be expected from a CNN 

CNN architecture comprises of a few convolutional layers,each followed by a pooling layer,
activation function and optionally batch normalization. An input moves throgh the network,it gets smaller,mostly because
of MaxPooling. The final layer outputs the class probabilities prediction. 

"""
# !/usr/bin/python

import os
import numpy
import tensorflow as tf
from dataclasses import dataclass
import tensorflow.keras as keras
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

@dataclass
class Hyperparameters:
    CLASSES = 10
    INPUT_SHAPE = (28,28,1)
    BATCH_SIZE = 200
    EPOCHS = 10
    LOGS_DIR = "/tmp/tb/tf_logs/"

config = Hyperparameters()


(xTrain,yTrain),(xTest,yTest) = tf.keras.datasets.mnist.load_data()

xTrain = xTrain.astype('float32')/255
xTest  = xTest.astype('float32')/255

xTrain = numpy.expand_dims(xTrain,-1)
xTest  = numpy.expand_dims(xTest,-1)

#Debug: print(xTrain.shape,yTrain.shape)

yTrain = tf.keras.utils.to_categorical(yTrain,config.CLASSES)
yTest = tf.keras.utils.to_categorical(yTest,config.CLASSES)


model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape= config.INPUT_SHAPE),
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(padding='same'),
    tf.keras.layers.Conv2D(64,kernel_size=(1,1),activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(config.CLASSES,activation='softmax')

])

#Debug: model.summary()
tBoardCallback = keras.callbacks.TensorBoard(config.LOGS_DIR,histogram_freq = 1, profile_batch = (500,520))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(xTrain,yTrain,batch_size=config.BATCH_SIZE ,epochs=config.EPOCHS,callbacks = [tBoardCallback])

loss,accuracy = model.evaluate(xTest,yTest,verbose=0)

print("Loss:",loss,"Accuracy:",accuracy)

model.save("Conv2D_MNIST.h5")
model.save_weights("Conv2D_MNIST_Weights.h5")
