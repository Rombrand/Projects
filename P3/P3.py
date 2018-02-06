import os

import tensorflow as tf
#import numpy as np
import csv
#import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, core, Lambda, Cropping2D
#import sklearn
#from sklearn.model_selection import train_test_split
#from random import shuffle

import matplotlib.pyplot as plt
#import helpers

from helpers import *


#PATH = 'data/'
#PATH = 'data_real_training/'

#lines = []
samples = []
with open(PATH + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #lines.append(line)
        samples.append(line)
#del(samples[0])

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


train_generator = generator(samples, 256)

for i in range(3):
    X_batch, y_batch = next(generator(samples))
    print(X_batch.shape, y_batch.shape)

#
# images = []
# measurements = []
# for line in lines:
#     source_path = line[0]        # first element is the central image
#     filename = source_path.split('/')[-1]   # last / is the filename (picture)
#     #filename = img_file.split('/')[-1]
#     #filename = 'data/IMG/' + filename.split('\\')[-1]
#     current_path = PATH + 'IMG/' + filename
#     image = cv2.imread(current_path)
#     images.append(image)
#
# #    if image == None:
#  #       print("incorrect image path or missing image: ", current_path)
#
#     measurement = float(line[3])    # 4th token is the steering angle
#     measurements.append(measurement)
#
#     X_train = np.array(images)
#     y_train = np.array(measurements)

#-------



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)






#############



def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (160, 80), cv2.INTER_AREA)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
#model.add(Lambda(resize(image)))

#model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(Convolution2D(32, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(core.Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))








"""

model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
    args.samples_per_epoch,
    args.nb_epoch,
    max_q_size=1,
    validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
    nb_val_samples=len(X_valid),
    callbacks=[checkpoint],
    verbose=1)


model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3, verbose = 1)


model.fit(X_train, y_train, validation_split= 0.2, shuffle = True, nb_epoch = 7)

model.save('model.h5')

"""

model.compile(loss='mse', optimizer='adam')


history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)*4, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=7, verbose = 1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
