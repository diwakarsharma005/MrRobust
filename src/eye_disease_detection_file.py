"""Keras code for Eye Disease Detection"""

# The dataset (1.67 GB) can be downloaded from https://www.kaggle.com/jr2ngb/cataractdataset .


# Necessaryy libraries for the Neural Network

import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from IPython.display import SVG
from IPython.display import Image
from keras.utils import plot_model


import warnings
warnings.filterwarnings('ignore')





# Making an object of sequential class that initializes the CNN
eye_NN = Sequential()

# Adding a convolutional layer that takes input image of size 150X150
# This layer contains 64 filters of kernel size 3 each
# Activation function for the neurons or nodes in this layer is 'relu'
eye_NN.add(Convolution2D(64, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))

# Max Pooling layer to extarct the prominent features from the images which helps in better prediction of the images
eye_NN.add(MaxPooling2D(pool_size = (2, 2)))

# The second convolution layer does exactly what the firs layer does 
# Adding another layer helps fine tune the model even further
# This layer contains 64 filters of kernel size 3 each
eye_NN.add(Convolution2D(64, 3, 3, activation = 'relu'))

# Also another max pooling layer to extract the more prominent features from the images that passes through the 2nd convolution layer
eye_NN.add(MaxPooling2D(pool_size = (2, 2)))

# This is the third and the last convolutional layer which has 32 filters
eye_NN.add(Convolution2D(32, 3, 3, activation = 'relu'))

#Another max pooling layer helps in increasing the overall accuracy of the model
eye_NN.add(MaxPooling2D(pool_size = (2, 2)))


# The flatten layer is absolutely essential which flattens the matrix of numbers representing the filtered and pooled images into a single row of numbers

eye_NN.add(Flatten())

# In this layer we establish the full connection with all the layers of the neural network
#This output layer contains 256 nodes which will further help in increasing the accuracy of the model
eye_NN.add(Dense(output_dim = 256, activation = 'relu'))

#This dense layer outputs the classes which we are trying to represent in the form of a single digit
eye_NN.add(Dense(output_dim = 4, activation = 'softmax'))

# Comiling the neural network with 'adam' optimizer
eye_NN.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])




from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)   
# rescale  will rescale the pixel values from 0-255 to 0-1 which makes training faster 
# shear_range will randomly shear images from different sides
# zoom_range will randomly zoom the images
# horizontal_filp will randomly flip half of the images horizontally

test_datagen = ImageDataGenerator(rescale = 1./255)
# the reson we dont apply the other three parameteres is beacuse that will dpend upon the user input




# training_set comtains 450 images from 4 classes of images 1.normal 2.cataract 3.glaucoma 4.Retina Disease
# batch_size of 10 specifies the gap after which the model gets updated
training_set = train_datagen.flow_from_directory('final/train',target_size = (150, 150),batch_size = 10, class_mode = 'categorical')

# test_set comtains 150 images from 4 classes of images 1.normal 2.cataract 3.glaucoma 4.Retina Disease
test_set = test_datagen.flow_from_directory('final/test', target_size = (150, 150), batch_size = 10, class_mode = 'categorical')                                    




# training the model on the training and test images
#samples per epoch determine number of images passing each epoch
# nb_epoch specifies the number of times the network passes through the dataset
# nb_val_samples determine the number of samples on which the model is validated
eye_NN.fit_generator(training_set, samples_per_epoch = 450, nb_epoch = 120, validation_data = test_set, nb_val_samples =100 )




user_image = image.load_img('final_1/test/1_normal/NL_008.png', target_size = (150, 150)) # taking the user input for evaluation
 
user_image = image.img_to_array(user_image)                                          # flattening the image into numy array

user_image = np.expand_dims(user_image, axis = 0)                                    # adds a fake dimension because predict function expects batch of images
 
output = eye_NN.predict_classes(user_image)                                      # calling the predict method

print("The user input image :")
Image(filename='final_1/test/1_normal/NL_008.png',  width = 150, height = 80) # Loading image in output that the user wants to see




print("Class preditction is", output)

if output== 0:
    prediction = 'normal eye'
    print("Congratulations, you have a perfectly ",prediction)

elif output == 1:
    prediction = 'cataract'
    print("It seems that you are suffering from ",prediction)    

elif output == 2:
    prediction = 'Glaucoma'
    print("It seems that you are suffering from ",prediction)    
     
else:
    prediction = 'Retina Disease'
    print("It seems that you are suffering from  ",prediction)


# For printing a string summary of the network
eye_NN.summary()

# For outputting Keras Graph
%show_model eye_NN