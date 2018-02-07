import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from params import *

#image = cv2.imread('test_images/warp_img_with_lines.jpg') #straight_lines1  test5
#test_image = cv2.imread('test_images/test5.jpg') #straight_lines1  test5
#test_image = cv2.imread('test_images/straight_lines1.jpg') #straight_lines1  test5






"""

# LONG
# Define Source and Destination points
x1_offset = 150
x2_offset = 590     #570
x3_offset = (x2_offset - x1_offset)/3
y_offset = size_y-270   # 250
# Source points
P1_src = (x1_offset, size_y)
P2_src = (x2_offset, y_offset)
P3_src = (size_x-x2_offset, y_offset)
P4_src = (size_x-x1_offset, size_y)
# Destination points
P1_dst = (P1_src[0] + x3_offset, size_y)
P2_dst = (P1_dst[0], 0) # y_offset)
P3_dst = (P4_src[0]-x3_offset, 0) #y_offset)
P4_dst = (P3_dst[0], size_y)




# SHORT
# Define Source and Destination points
x1_offset = 180
x2_offset = 588
x3_offset = (x2_offset - x1_offset)/3
y1_offset = 30
y2_offset = size_y-270   # 250
# Source points
P1_src = (x1_offset, size_y-y1_offset)
P2_src = (x2_offset, y2_offset)
P3_src = (size_x-x2_offset, y2_offset)
P4_src = (size_x-x1_offset, size_y-y1_offset)
# Destination points
P1_dst = (P1_src[0] + x3_offset, size_y)
P2_dst = (P1_dst[0], 0) # y_offset)
P3_dst = (P4_src[0]-x3_offset, 0) #y_offset)
P4_dst = (P3_dst[0], size_y)

src = np.float32([P1_src, P2_src, P3_src, P4_src])
dst = np.float32([P1_dst, P2_dst, P3_dst, P4_dst])
"""







#-------------------------- CALIBRATION ------------------------
def calibration_params():

    obj_points = []  # Real world points
    img_points = []  # Image points

    # Prepare obj points
    objp = np.zeros((nx * ny, 3), np.float32)
    # generate x/y-coordinates (x,y,z=0)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Read in an image
    images = glob.glob('camera_cal/calibration*.jpg')
    print(images)

    for image in images:

        img = cv2.imread(image)
        print(img.shape)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        print(ret)
        if ret == True:
            #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)
            obj_points.append(objp)
            img_points.append(corners)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    img = cv2.imread('camera_cal/calibration1.jpg')
    #image = cv2.imread(images[0])
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)


    # Plot image to test undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(undist_img)
    ax2.set_title('Undistorted Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

    # Saving params:
    with open('calibration_params.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([mtx, dist], f)

    return mtx, dist


#-------------------------- UNDISTORT ------------------------
def undistort(image, mtx, dist):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist


#----------------------------- WARP --------------------------
def warp(undist, src, dst):

        # get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # warp the image to a top-down view
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
        #cv2.polylines(warped, vertices_src, False, (0, 255, 0), 2)
        return warped


"""


undist_img = undistort(test_image, mtx, dist)

warp_img = warp(undist_img, src, dst)

#cv2.imwrite('../test_images/warp_curved_lines.png',warp_img)


# plot images
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(undist_img)
ax2.set_title('Undistorted Image', fontsize=20)
ax3.imshow(warp_img)
ax3.set_title('Warped Image', fontsize=20)
ax4.imshow(img_src)
ax4.set_title('Source Points', fontsize=20)
ax5.imshow(img_dst)
ax5.set_title('Destination Points', fontsize=20)


plt.show()









    





# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()


# Read in an image
img = cv2.imread('test_image2.png')


""

# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]
    size_x, size_y = img_size
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    for i, corner in enumerate(corners):
        print("corner {}: {}".format(i, corner))
    # 4) If corners found:
    if ret == True:
        # a) draw corners
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # plt.imshow(img)

        src = np.float32([corners[0], corners[nx-1], corners[-nx], corners[-1]])
        print("SRC: ", src)
        # arbitrary destination points
        xy_offset = 80  # choosen distance in x and y direction
        dst = np.float32([[xy_offset, xy_offset], [size_x - xy_offset, xy_offset], [xy_offset, size_y - xy_offset], [size_x - xy_offset, xy_offset]])
        # get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # warp the image to a top-down view
        print("Image size = ", img_size)
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M

"""
