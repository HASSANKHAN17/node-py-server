# =============================================================================
# # Convolutional Neural Network
# 
# # Importing the libraries
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
# tf.__version__
# 
# # Part 1 - Data Preprocessing
# 
# # Preprocessing the Training set
# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)
# training_set = train_datagen.flow_from_directory('dataset/training_set',
#                                                  target_size = (64, 64),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')
# 
# # Preprocessing the Test set
# test_datagen = ImageDataGenerator(rescale = 1./255)
# test_set = test_datagen.flow_from_directory('dataset/test_set',
#                                             target_size = (64, 64),
#                                             batch_size = 32,
#                                             class_mode = 'binary')
# 
# # Part 2 - Building the CNN
# 
# # Initialising the CNN
# cnn = tf.keras.models.Sequential()
# 
# # Step 1 - Convolution
# cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# 
# # Step 2 - Pooling
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# 
# # Adding a second convolutional layer
# cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# 
# # Step 3 - Flattening
# cnn.add(tf.keras.layers.Flatten())
# 
# # Step 4 - Full Connection
# cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# 
# # Step 5 - Output Layer
# cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# 
# # Part 3 - Training the CNN
# 
# # Compiling the CNN
# cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# 
# # Training the CNN on the Training set and evaluating it on the Test set
# cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
# 
# # Part 4 - Making a single prediction
# 
# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = cnn.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'dog'
# else:
#     prediction = 'cat'
# print(prediction)
# =============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow as tf
import shutil,os
import requests
from keras.preprocessing.image import ImageDataGenerator
from string import Template
link = Template('$x')
# print(__version__)

#preprocssin... augmentation

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')  #number of classes

#test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


#building the cnn

cnn=keras.models.Sequential()
#cnnnovoliutujrjoijrn
cnn.add(keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",input_shape=[64,64,3]))

#pooling
cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))  #pool size represents shifting the squares
#adding a second convilutional layer
cnn.add(keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"))
cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

#flattening
cnn.add(keras.layers.Flatten())

#full connection
cnn.add(keras.layers.Dense(units=128,activation="relu"))

#output layer
cnn.add(keras.layers.Dense(units=1,activation="sigmoid")) #1 neuron for classfication ...use softmax for multiple classfication


#------------------------------------------------------------------------------training cnn
#compiling the cnn

cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#training the cnn on the training set and evaluating it on the test set

cnn.fit(x=train_set,validation_data=test_set, epochs=15)

#making  a asingle prediction
from keras.preprocessing import image

test_image=image.load_img("uploads/photo.jpg",target_size=(64,64))
#convert pale to array...convert pale to array..which is in pil format
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0) #first dimension of batch
result=cnn.predict(test_image)
print(result)
train_set.class_indices
if(result[0][0]==1):
    prediction="suspicious"
#     shutil.move("")
else:
    prediction="relatives"

print(prediction)
data = prediction

























