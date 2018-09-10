# Tensorflow Image Classification (Fruit Classification)


## Overview
This is a Deep Neural Network that classifies 15 different kinds of fruit by using Keras and Tensorflow. 

![](https://www.livemint.com/rf/Image-621x414/LiveMint/Period2/2017/10/31/Photos/Processed/fruits-kFLF--621x414@LiveMint.jpg)

The following are the different kinds of fruit that are classified by this Neural Network.
1. Apple
1. Avocado
1. Banana
1. Cherry
1. Dates
1. Guava
1. Mango
1. Mulberry
1. Orange
1. Papaya
1. Pear
1. Pineapple
1. Plum
1. Pomegranate
1. Strawberry


## Convolutional Neural Network (CNN)
Convolutional Neural Network is a branch of Machine Learning,  most commonly used to analyze and predict visual imagery. 
There are several layers in CNN that include Convolution layer, Pooling layer, Flatten layer, Fully Connected layer. The fully connected layer classifies the fruit.

![](https://cdn-images-1.medium.com/max/1000/1*NQQiyYqJJj4PSYAeWvxutg.png)


## Implementation

### Importing necessary libraries
The keras library needs to be installed in order to import classes from it.

`from keras.models import Sequential`

Sequential class is used to initialize the Convolution Neural Network.

`from keras.layers import Convolution2D`

Convolution2D class creates Convolutional Layers from the images

`from keras.layers import MaxPooling2D`

MaxPooling 2D class creates pools from the Convolution Layer

`from keras.layers import Flatten`

The Flatten class flattens the pooled values into a single vector, that serves as input for the Neural Network.

`from keras.layers import Dense`

Dense function creates a fully connected Neural Network.

`from keras.preprocessing.image import ImageDataGenerator`

The ImageDataGenrator class from keras.preprocessing.image library performs several operations on images.

`from keras.preprocessing import image as img`

This class is used for testing the classification on a new image.

`from keras.models import load_model`

This class is used to save trained model

`import numpy as np`

numpy is used process arrays


### Building CNN

#### Step 1:

`classifier = Sequential()`

This Initializes an object classifier using the Sequential class. The Sequential class also initializes the neural network.

#### Step 2:

`classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))`

Using data of all the pixels of an image is computationally expensive, so we are using convolutional layer to reduce input features. (32,3,3) represent number of convolution filters, number of rows in each convolution kernel and number of columns in each convolution kernel. input_shape = (64,64,3) represent height,width and depth of input image. The activation function used is 'relu' (rectifier).

![](https://i.stack.imgur.com/I7DBr.gif)

#### Step 3:

`classifier.add(MaxPooling2D(pool_size = (2,2)))`

Using maxpooling further reduces the features. pool_size = (2,2) represent sliding a 2x2 filter over the previous layer, taking a maximum of 4 values.

![](http://ufldl.stanford.edu/wiki/images/0/08/Pooling_schematic.gif)

#### Step 4:

`classifier.add(Convolution2D(32,3,3,activation='relu'))`

`classifier.add(MaxPooling2D(pool_size = (2,2)))`

We are performing using Convolution layer and Maxpooling layer for the second time. This is called deep learning.

#### Step 5:

`classifier.add(Flatten())`

The flatten class is used to convert the maxpool layer to a single-dimensional vector that is passed as input to the fully connected Neural Network.

#### Step 6:

`classifier.add(Dense(output_dim=128, activation = 'relu'))`

`classifier.add(Dense(output_dim=15, activation = 'softmax'))`

A fully connected Convolutional Neural Network is now created. SInce the number of dimensions in the input layer is unknown, an adequate value for output_dim is used. The final layer has output_dim as 15 which corresponds to 15 different outcomes of the Neural Network. And the activation function used is softmax.

#### Step 7:

`classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])`

The complete neural network is now compiled. loss = categorical_crossentropy is used since there are more than two outcomes. And we are evaluating the performance based on accuracy.

### Fitting Convolutional Neural Network to Images

`train_datagen = ImageDataGenerator(`

`rescale=1./255,`

`shear_range=0.2,`
 
`zoom_range=0.2,`
     
`horizontal_flip=True)`



Several operations like rescale, shear, zoom, horizontal flip are performed on images. This eases the classification of images.




`test_datagen = ImageDataGenerator(rescale=1./255)`

`training_set = train_datagen.flow_from_directory('dataset/training_set',`
                                                    
`target_size=(64, 64),`
                                                    
`batch_size=32,``
                                                    
`class_mode='categorical')`



Directory for training set, target size, batch size and class mode are mentioned here. Tensorflow expects training set to 
contain images of different fruits in separate folders. Target size is nothing but the size of input images.



`test_set = test_datagen.flow_from_directory('dataset/test_set',`

`target_size=(64, 64),`

`batch_size=32,`

`class_mode='categorical')`



Directory for test set, target size, batch size and class mode are mentioned here. Tensorflow expects test set to 
contain images of different fruits in separate folders. Target size is nothing but the size of input images.



`classifier.fit_generator(training_set,`

`steps_per_epoch=7247,`

`epochs=5,`

`validation_data=test_set,`

`validation_steps=4364)`


There is a total of 7247 images in the training set and 4364 images in the test set. Training a Convolution Neural Network is computationally intensive, hence epochs = 5




![](https://lh3.googleusercontent.com/lAVECxvSGcHDLiQ9EmAUPYAtvXIQ94A6qUeBjw14rkJE-eGwHsTqKhCGgceuHGoLKngyi0oZy4wwkfjYoOrEqwLkhLdn2JEJTS4-w6jgcnuT8lCHGXenN-5BjWvzaBw-JiHd1R7QN-KgDkLiwPcA7H8F2isIyJx0g7lOq8N2x9CumLhE4nWugM2d0Gms_6ihAqpBlGRUCbUnIuUOkL98wE6X1NT2arae2mkjcKwGd_LYUfeLJo4V7wHc-S_3pccpYfOdJZxeEND-g4CYWfce0j0Q2fAGG8W6QwUWP_Avjqzh-aAI_ZaN82xWk3vcuIFWQ7NjYoiXAsVYYA8X1I4JAgneqFvUv7-EpXvU0j8Vazg7dLHZ6rlJiW1LIJsOnFY-K3HNsS3_nEGbbn0pThgi4YvBGfoNvqgSZIXt_DjlI0tP5WxA_2DnA0fdQiYzfzNfqlfDxn85hmDV1mSDdTV_UIyepXqumMB28Mih94O3VDf0FCG5X2R3diH3MZvkjx7Qq1zvH5xC2wtgyJueYx6ehBbOMbMeUAhU5dZ27A67HRE0gPix3KMHi-gDo1Bo37acZO42EO1iPF4dd0p_TerdRb6qQsOqfxwEq9aGT8AhVQSJXs3HLmH7klXrvXqamD8=w613-h190-no)


![](https://lh3.googleusercontent.com/Ffk_Cf74nPRqUICQ7MeF7gePt2ouyLmRrWuzJA5R0NzY429_E9M4RvvlRdhDHwo5ue47HgFpmM1tryzQS5uct0pd6f3DLkN3tFOdPs-AFhTEhAQ98n1KAzit1pBj_eDk50HiDz3PpOdIY3NwNl7NHRYLzB6s0VPIGYBhgG5vVrYaXnYclehJo10PzY6_m-EY4qxUzOGT6fleS35YoT_pHbaweuOuUmNVjxffVTZrvJPXsEHTwhvoMdQa97ueUHuD2zOQme83ivhzUnNMMKu7EbAf_tnm6BXnq8XyQCMGF3SX_GWjbl4E3LzeBkMRh4N4tNwdLMdV7Rv-EbWQNs7fA_M3FmYaqY9kKV0gGEoh4p8LHyOOcGJNHjuINLj5WDWjJxi6FylKp-c7aKQQpj7uqjmrM4Ajr3cVaUmxOjubLt5d8N7iejNJLOyxC172TOPMwBVcGBY0mkBzYKddvMtW653LHkjv-hzsR3Kqjrt_HkHyLX45S_zQgKlfqf_jfifxoM-9VxzBs_7JJGITMPJhD-00Bqpq0-Kqu1UaMX67NzAvsSmQAcsRgANin7iz7HACpKThiZJMoz2Bg6SNJ_d4MnHdkYFZbDgpqchTcM4phiE-LBS__cySlWqPKgW2qoo=w606-h320-no)


### Classifying a new picture

`test_image = img.load_img('dataset/new_set/pineapple.jpg',`
                          
`target_size=(64, 64))`

`test_image = img.img_to_array(test_image)`

`test_image = np.expand_dims(test_image,axis = 0)`

`result = classifier.predict_on_batch(test_image)`


New images can be tested by creating a new folder(new_set) in the root folder that contains training set and test set. The object 'result' is a 1x15 matrix that has binary output (0 or 1). Testing an image in the new_set gives binary value 1 in one of the fifteen columns. The column with value 1 corresponds to that particular fruit. 

For instance, if value of result[0,4] = 1, the value corresponds to the 5th fruit i.e, Dates.


### Creating if conditions that gives the name of predicted fruit


`if result[0,0] == 1:`
    `fruit = 'Apple'`

`elif result[0,1] == 1:`
     `fruit = 'Avocado'`

`elif result[0,2] == 1:`
     `fruit = 'Banana'`

`elif result[0,3] == 1:`
     `fruit = 'Cherry'`

`elif result[0,4] == 1:`
     `fruit = 'Dates'`

`elif result[0,5] == 1:`
     `fruit = 'Guava'`

`elif result[0,6] == 1:`
     `fruit = 'Mango'`

`elif result[0,7] == 1:`
     `fruit = 'Mulberry'`

`elif result[0,8] == 1:`
     `fruit = 'Orange'`

`elif result[0,9] == 1:`
    `fruit = 'Papaya'`

`elif result[0,10] == 1:`
    `fruit = 'Pear'`

`elif result[0,11] == 1:`
    `fruit = 'Pineapple'`

`elif result[0,12] == 1:`
    `fruit = 'Plum'`

`elif result[0,13] == 1:`
    `fruit = 'Pomegranate'`

`else: fruit = 'Strawberry'`


![](https://lh3.googleusercontent.com/i_v1gEh9TORCcoZgLMU-wpgJimehEkvL5rMkC4xOCGGns1AayA0SSrQIXXxMXOT5xloA6hA0ME5QCe5RhcIP0m5DZv17ld-3q074NehN26PkeJoZBtyY0fgivNvUzceELNmKY7F9oY5Xhh12s-uKj3dw27438NxEXsx6VKrwjCZ-Sxp6KOBjHd9lEBYIRYhw8BgvpPXGwRbcGUSmIcFMHWfqhs3aEi5C35QessWjz6ej2erVF1mqqHCF2bqOq5HeL58jtiD-1Xhjlf3zfqKmAZ57aRTxB-6l-69zln8Rg3R70AA2Iu9Xgggbs_6_1-AYEMRNQiTPLbf7Efud38F0RBsj0EKAB9Yrfb25XdBBn3UGDUaVN3-iXh8CRNiLc-pkBu-ZmuIvNc7rJEDkAR9uA9Wt0kb5xrXfrxCUBLx0kJ4OUbyhs-5YrP07WyVzVMTk2naXwWhe6DJ08su6_F2riMAbWejurqKmQNxTGwvsr_N1jX4FX-QZb2nCwuOhvEziW5oyy6EG0JtmM6bj4SZ0eKu6PnrprtEOKL9YjavfNiFtE2IeQiNqLs5y9guuyqBotfzS_UuZIOEJiuPQ257CzRhlJDnvyo-vVpW2OIiRIAhzOtzFDBkgxYGH-cImgjI=w1260-h827-no)
### Saving the trained model


`classifier.save('CNN.h5')  `

`model = load_model('CNN.h5')`


