import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
#import pickle
from sklearn.externals import joblib
from helpers import *
from parameters import *


#--------------------------------------------- TRAINING SEQUENCE -------------------------------------------------------

#------------------------------------------ Import Training Images -----------------------------------------------------
if train == True:
    t = time.time()

    cars = []
    notcars = []
    # cars
    car_images = glob.glob('../../Project_Data/P5_data/big/vehicles/**/*.png', recursive=True)
    #car_images = glob.glob('../../Project_Data/P5_data/small/vehicles/**/*.jpeg', recursive=True)
    for image in car_images:
        cars.append(image)
        #print("car")

    # non-cars
    notcar_images = glob.glob('../../Project_Data/P5_data/big/non-vehicles/**/*.png', recursive=True)
    #notcar_images = glob.glob('../../Project_Data/P5_data/small/non-vehicles/**/*.jpeg', recursive=True)

    for image in notcar_images:
        notcars.append(image)
        #print("non_car")

    no_of_images = [len(cars), len(notcars)]
    print("# of Images: ", no_of_images)


#------------------------------------------- Extract Car Features ------------------------------------------------------

    # Create a list to append feature vectors to
    car_feature_list = []
    # Iterate through the list of images
    print("Extracting car features...")
    for image_file in car_images:
        # Read in each one by one
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        feature = extract_features(image, color_space=color_space, spatial_size=spatial_size,
                         hist_bins=hist_bins, orient=orient,
                         pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                         spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, block_norm=block_norm)

        car_feature_list.append(feature)
    print("# of car features: ", len(car_feature_list))


#------------------------------------------- Extract NotCar Features ------------------------------------------------------
    notcar_feature_list = []
    # Create a list to append feature vectors to
    feature_list = []
    # Iterate through the list of images
    print("Extracting noncar features...")
    for image_file in notcar_images:
        # Read in each one by one
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        feature = extract_features(image, color_space=color_space, spatial_size=spatial_size,
                                           hist_bins=hist_bins, orient=orient,
                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                           spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, block_norm=block_norm)
        notcar_feature_list.append(feature)
    print("# of notcar features: ", len(notcar_feature_list))

# --------------------------------------------- Stack Features ---------------------------------------------------------
    # Create an array stack of feature vectors
    X = np.vstack((car_feature_list, notcar_feature_list)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_feature_list)), np.zeros(len(notcar_feature_list))))

# --------------------------------------------- Split Data ---------------------------------------------------------
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# --------------------------------------------- Scale Data ---------------------------------------------------------
    print('Features in X_train:', len(X_train[0]))
    print("Scaling data...")
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    joblib.dump(X_scaler, 'X_scaler.pkl')


    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

# ---------------------------------------- Create and Apply Classifier -------------------------------------------------
    print("Training classifier...")
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t3 = time.time()
    svc.fit(X_train, y_train)
    t4 = time.time()
    # store classifier
    joblib.dump(svc, 'svc_classifier.pkl')

    print(round(t4 - t3, 2), 'Seconds to train SVC...')

# ----------------------------------------- Test The Classifier ---------------------------------------------------------
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t5 = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t6 = time.time()
    print(round(t6 - t5, 5), 'Seconds to predict', n_predict, 'labels with SVC')


    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train the classifier on {} pictures'.format(len(y)))
else:
    # load classifier
    svc = joblib.load('svc_classifier.pkl')
    X_scaler = joblib.load('X_scaler.pkl')


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- CAR DETECTION PIPELINE -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

image_file = glob.glob('../../Project_Data/P5_data/small/*.jpeg', recursive=True)
print(image_file)
image = cv2.imread(image_file[1])
image = cv2.imread('../CarND-Vehicle-Detection/test_images/test1.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Scale: ", np.amax(image))


"""

image_fatures = extract_features(image, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=hist_feat, hist_feat=hist_feat, hog_feat=hog_feat, block_norm=block_norm)
"""
#print('Feature vector length:', len(image_fatures))


# Fit a per-column scaler
# Apply the scaler to X
#scaled_feature = X_scaler.transform(np.array(image_fatures).reshape(1, -1))



#print('My SVC predicts: ', svc.predict(scaled_feature))





out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

plt.imshow(out_img)
plt.show()


######
exit()
######