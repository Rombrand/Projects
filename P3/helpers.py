import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle
import cv2
import numpy as np


PATH = 'data_real_training_2/'
#PATH = 'data/'

correction = 0.3
#
# XL_train = np.copy(X_train)
#
# XL_train = np.asarray(list(map(flip, XL_train)))
#
# X_train = np.vstack((X_train, XL_train))
#
# X_train = np.vstack((X_train, XR_train))
# print("SHAPE_R: ", X_train.shape)

def flip(img, angle):
    #print("\nimg+angle\n")

    img = cv2.flip(img, 1)
    angle *= -1
    return img, angle



def generator(samples, batch_size=256):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = PATH + 'IMG/'+batch_sample[0].split('/')[-1] # first token is the central image, last / is the filename (picture)
                center_image = cv2.imread(name)

                name = PATH + 'IMG/'+batch_sample[1].split('/')[-1] # first token is the central image, last / is the filename (picture)
                left_image = cv2.imread(name)

                name = PATH + 'IMG/'+batch_sample[2].split('/')[-1] # first token is the central image, last / is the filename (picture)
                right_image = cv2.imread(name)

                #print("BATCH[0]: ", batch_sample[0])

                center_angle = float(batch_sample[3])               # 4th element is the steering angle
                images.append(center_image)
                angles.append(center_angle)

                left_angle = center_angle + correction
                right_angle = center_angle - correction

                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

                flipped_image, flipped_angle = flip(center_image, center_angle)
                images.append(flipped_image)
                angles.append(flipped_angle)



            X_train = np.array(images)
            y_train = np.array(angles)

            #print("\nPath:  \n", batch_sample)
            #print("\nFilename: \n", name)

            yield sklearn.utils.shuffle(X_train, y_train)

