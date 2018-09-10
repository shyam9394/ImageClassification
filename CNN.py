#Convolutional Neural Network(CNN)


#Section 1 :Building the CNN

#Importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Step 1: Initializing the CNN
classifier = Sequential()

#Step 2 : Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#Step 3: MaxPooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 4: Performing Convolution and MaxPooling for the second time
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 5:Flatenning
classifier.add(Flatten())

#Step 6: Full Connection
classifier.add(Dense(output_dim=128, activation = 'relu'))
classifier.add(Dense(output_dim=15, activation = 'softmax'))
#Step 7: Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])

#Section 2: Fitting CNN to Images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                            steps_per_epoch=7247,
                            epochs=5,
                            validation_data=test_set,
                            validation_steps=4364)

#Section 3: Classifying a new picture
from keras.preprocessing import image as img
import numpy as np

test_image = img.load_img('dataset/new_set/pineapple.jpg',
                          target_size=(64, 64))
test_image = img.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = classifier.predict_on_batch(test_image)

#Creating a st
if result[0,0] == 1:
    fruit = 'Apple'
elif result[0,1] == 1:
     fruit = 'Avocado'
elif result[0,2] == 1:
     fruit = 'Banana'
elif result[0,3] == 1:
     fruit = 'Cherry'
elif result[0,4] == 1:
     fruit = 'Dates'
elif result[0,5] == 1:
     fruit = 'Guava'
elif result[0,6] == 1:
     fruit = 'Mango'
elif result[0,7] == 1:
     fruit = 'Mulberry'
elif result[0,8] == 1:
     fruit = 'Orange'
elif result[0,9] == 1:
    fruit = 'Papaya'
elif result[0,10] == 1:
    fruit = 'Pear'
elif result[0,11] == 1:
    fruit = 'Pineapple'
elif result[0,12] == 1:
    fruit = 'Plum'
elif result[0,13] == 1:
    fruit = 'Pomegranate'
else: fruit = 'Strawberry'

   

#Saving the trained model
from keras.models import load_model

classifier.save('CNN.h5')  

model = load_model('CNN.h5')




