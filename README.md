Tensorflow Image Classification (Fruit Classification)
Overview
This is a Deep Neural Network that classifies 15 different kinds of fruit by using Keras and Tensorflow.



The following are the different kinds of fruit that are classified by this Neural Network.

Apple
Avocado
Banana
Cherry
Dates
Guava
Mango
Mulberry
Orange
Papaya
Pear
Pineapple
Plum
Pomegranate
Strawberry
Convolutional Neural Network (CNN)
Convolutional Neural Network is a branch of Machine Learning, most commonly used to analyze and predict visual imagery. There are several layers in CNN that include Convolution layer, Pooling layer, Flatten layer, Fully Connected layer. The fully connected layer classifies the fruit.



Implementation
Importing necessary libraries
The keras library needs to be installed in order to import classes from it.

from keras.models import Sequential

Sequential class is used to initialize the Convolution Neural Network.

from keras.layers import Convolution2D

Convolution2D class creates Convolutional Layers from the images

from keras.layers import MaxPooling2D

MaxPooling 2D class creates pools from the Convolution Layer

from keras.layers import Flatten

The Flatten class flattens the pooled values into a single vector, that serves as input for the Neural Network.

from keras.layers import Dense

Dense function creates a fully connected Neural Network.

from keras.preprocessing.image import ImageDataGenerator

The ImageDataGenrator class from keras.preprocessing.image library performs several operations on images.

from keras.preprocessing import image as img

This class is used for testing the classification on a new image.

from keras.models import load_model

This class is used to save trained model

import numpy as np

numpy is used process arrays

Building CNN
Step 1:
classifier = Sequential()

This Initializes an object classifier using the Sequential class. The Sequential class also initializes the neural network.

Step 2:
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

Using data of all the pixels of an image is computationally expensive, so we are using convolutional layer to reduce input features. (32,3,3) represent number of convolution filters, number of rows in each convolution kernel and number of columns in each convolution kernel. input_shape = (64,64,3) represent height,width and depth of input image. The activation function used is 'relu' (rectifier).



Step 3:
classifier.add(MaxPooling2D(pool_size = (2,2)))

Using maxpooling further reduces the features. pool_size = (2,2) represent sliding a 2x2 filter over the previous layer, taking a maximum of 4 values.



Step 4:
classifier.add(Convolution2D(32,3,3,activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

We are performing using Convolution layer and Maxpooling layer for the second time. This is called deep learning.

Step 5:
classifier.add(Flatten())

The flatten class is used to convert the maxpool layer to a single-dimensional vector that is passed as input to the fully connected Neural Network.

Step 6:
classifier.add(Dense(output_dim=128, activation = 'relu'))

classifier.add(Dense(output_dim=15, activation = 'softmax'))

A fully connected Convolutional Neural Network is now created. SInce the number of dimensions in the input layer is unknown, an adequate value for output_dim is used. The final layer has output_dim as 15 which corresponds to 15 different outcomes of the Neural Network. And the activation function used is softmax.

Step 7:
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])

The complete neural network is now compiled. loss = categorical_crossentropy is used since there are more than two outcomes. And we are evaluating the performance based on accuracy.

Fitting Convolutional Neural Network to Images
train_datagen = ImageDataGenerator(

rescale=1./255,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True)

Several operations like rescale, shear, zoom, horizontal flip are performed on images. This eases the classification of images.

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',

target_size=(64, 64),

`batch_size=32,``

class_mode='categorical')

Directory for training set, target size, batch size and class mode are mentioned here. Tensorflow expects training set to contain images of different fruits in separate folders. Target size is nothing but the size of input images.

test_set = test_datagen.flow_from_directory('dataset/test_set',

target_size=(64, 64),

batch_size=32,

class_mode='categorical')

Directory for test set, target size, batch size and class mode are mentioned here. Tensorflow expects test set to contain images of different fruits in separate folders. Target size is nothing but the size of input images.

classifier.fit_generator(training_set,

steps_per_epoch=7247,

epochs=5,

validation_data=test_set,

validation_steps=4364)

There is a total of 7247 images in the training set and 4364 images in the test set. Training a Convolution Neural Network is computationally intensive, hence epochs = 5





Classifying a new picture
test_image = img.load_img('dataset/new_set/pineapple.jpg',

target_size=(64, 64))

test_image = img.img_to_array(test_image)

test_image = np.expand_dims(test_image,axis = 0)

result = classifier.predict_on_batch(test_image)

New images can be tested by creating a new folder(new_set) in the root folder that contains training set and test set. The object 'result' is a 1x15 matrix that has binary output (0 or 1). Testing an image in the new_set gives binary value 1 in one of the fifteen columns. The column with value 1 corresponds to that particular fruit.

For instance, if value of result[0,4] = 1, the value corresponds to the 5th fruit i.e, Dates.

Creating if conditions that gives the name of predicted fruit
if result[0,0] == 1: fruit = 'Apple'

elif result[0,1] == 1: fruit = 'Avocado'

elif result[0,2] == 1: fruit = 'Banana'

elif result[0,3] == 1: fruit = 'Cherry'

elif result[0,4] == 1: fruit = 'Dates'

elif result[0,5] == 1: fruit = 'Guava'

elif result[0,6] == 1: fruit = 'Mango'

elif result[0,7] == 1: fruit = 'Mulberry'

elif result[0,8] == 1: fruit = 'Orange'

elif result[0,9] == 1: fruit = 'Papaya'

elif result[0,10] == 1: fruit = 'Pear'

elif result[0,11] == 1: fruit = 'Pineapple'

elif result[0,12] == 1: fruit = 'Plum'

elif result[0,13] == 1: fruit = 'Pomegranate'

else: fruit = 'Strawberry'



Saving the trained model
classifier.save('CNN.h5')

model = load_model('CNN.h5')
