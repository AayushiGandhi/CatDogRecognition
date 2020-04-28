# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step1 = Convolution
#input_shape(rows,column,3 for color otherwise 1 for bnw) if tensorflow is running in backend
#else input_shape(3 for color otherwise 1 for bnw,column,rows) if theano is running in backend
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation = 'relu'))

#Step2 = Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step3 = Flattening
classifier.add(Flatten())

#Step4 = Full Connection
#output_dim = no of hidden nodes. Mostly use between no of input nodes and no of output nodes
classifier.add(Dense(output_dim = 128, activation = 'relu')) #for hidden layer i.e fully connected layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #for output layer
#sigmoid func for binary outcome. for more than two categories use softmax

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#for only 2 categories use binary_crossentropy and for more than 2 categories use categorial_crossentropy

#Fitting to CNN to the images
#Image augmentation
#If huge number of training set is not available then it becomes difficult for CNN to find corelations and patterns. Therefore the trick is to use image augmentation. It creates various batches of images and apply some sort of rotation and random transformations to same picture to make it look different
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
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)
