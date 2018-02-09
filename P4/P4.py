import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from line_detection import colour_extraction, gradient_extraction
from camera_calibration import undistort, warp, calibration_params
from measure_curvature import find_first_frame, find_lines, get_dist_radius, draw_lines
from params import *
from moviepy.editor import VideoFileClip



#----------------------------------------------------------------------
#---------------------------- PARAMETERS ------------------------------
#----------------------------------------------------------------------
print('start')


image = cv2.imread('test_images/prob1.png')  #straight_lines1  test5
#image = cv2.imread('test_images/test5.jpg') #straight_lines1  test5
#image = cv2.imread('test_images/straight_lines1.jpg') #straight_lines1  test5


# Convert to RGB
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



new_params = False  # Perform new calibration or use saved parameters
if new_params == True:
    mtx, dist = calibration_params()
else:
    # Loading calibration params:
    with open('calibration_params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        mtx, dist = pickle.load(f)





# Draw source and destination lines in images
"""
vertices_src = np.array([[P1_src, P2_src, P3_src, P4_src]], dtype=np.int32)
vertices_dst = np.array([[P1_dst, P2_dst, P3_dst, P4_dst]], dtype=np.int32)
#img_src = np.copy(test_image)
#img_dst = np.copy(test_image)
cv2.polylines(test_image, vertices_src, False, (0,255,0), 1)
cv2.polylines(test_image, vertices_dst, False, (255,0,0), 3)
plt.imshow(test_image)
plt.show()
"""

#exit()
#print("Type vert {}, \ntype src = {}".format(vertices_src, src))






#----------------------------------------------------------------------
#---------------------------- PIPELINE --------------------------------
#----------------------------------------------------------------------




def pipeline(image):
    #plt.imshow(image)
    #plt.show()
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(image)
    #plt.show()
    undist_img = undistort(image, mtx, dist)
    image = np.copy(undist_img)
    colour_bin = colour_extraction(image, r_thresh=r_thresh, s_thresh=s_thresh)
    gradient_bin = gradient_extraction(image, gauss_kernel=gauss_kernel, sobel_kernel=sobel_kernel,
                                       orient='x', thresh=thresh_gradient)

    combined_binaries = cv2.bitwise_or(colour_bin, gradient_bin)

    morphed = cv2.morphologyEx(combined_binaries, cv2.MORPH_OPEN, kernel=morph_kernel)
    warped_img = warp(morphed, src, dst)

    #sanity_check()

    if ((Left.detected == False) and (Right.detected == False)):
        find_first_frame(warped_img)
        # Saving params:
        # with open('found_lines.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        #   pickle.dump([left_fit, right_fit], f)
        print("first_")
    else:
        # Loading calibration params:
        # with open('found_lines.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
          #   left_fit, right_fit = pickle.load(f)
        find_lines(warped_img)
        print("second_")

    get_dist_radius()
    output_image = draw_lines(warped_img, undist_img)

    return output_image


#----------------------------------------------------------------------
#---------------------------- RUN --------------------------------
#----------------------------------------------------------------------
#pipeline(image)
#exit()

video_output = 'output_images/test.mp4'

clip1 = VideoFileClip("project_video.mp4")#.subclip(19,27)
result = clip1.fl_image(pipeline)
result.write_videofile(video_output, audio=False)


"""


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(3,5)
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!

%time white_clip.write_videofile(white_output, audio=False)



#plt.imshow(gradient_bin, cmap='gray')
#plt.show()



#undist_img = undistort(morphed, mtx, dist)



"""



"""
# Plot results line extraction
f, ((ax1, ax2, ax3), (bx1, bx2, bx3)) = plt.subplots(2, 3, figsize=(20, 8))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('original', fontsize=20)
ax2.imshow(colour_bin, cmap='gray')
ax2.set_title('Colour Binary', fontsize=20)
ax3.imshow(gradient_bin, cmap='gray')
ax3.set_title('Gradient Binary', fontsize=20)

bx1.imshow(combined_binaries, cmap='gray')
bx1.set_title('Combined: Colour-Gradient', fontsize=20)
bx2.imshow(morphed, cmap='gray')
bx2.set_title('Morphed', fontsize=20)
bx3.imshow(warped_img, cmap='gray')
bx3.set_title('Warped', fontsize=20)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

"""


#________________________________________________________________________
"""


# Read in an image and grayscale it
image = mpimg.imread('signs_vehicles_xygrad.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Calculate directional gradient
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        abs_sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sob= np.uint8(255*abs_sobel/np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sob)
    grad_binary[(scaled_sob >= thresh[0]) & (scaled_sob <= thresh[1])] = 1

    return grad_binary

# Calculate gradient magnitude
def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # Calculate the magnitude
    abs_sobel_xy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sob = np.uint8(255 * abs_sobel_xy / np.max(abs_sobel_xy))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sob)
    mag_binary[(scaled_sob >= mag_thresh[0]) & (scaled_sob <= mag_thresh[1])] = 1

    return mag_binary

# Calculate gradient direction
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # gradient in x and y separately
    sob_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sob_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_x = np.abs(sob_x)
    abs_y = np.abs(sob_y)
    # calculate the direction of the gradient
    gradient_direction = np.arctan2(abs_y, abs_x)
    #binary mask where direction thresholds are met
    dir_binary = np.zeros_like(gradient_direction)
    dir_binary[(gradient_direction >= thresh[0]) & (gradient_direction <= thresh[1])] = 1

    return dir_binary

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 80))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(30, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 255))
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

#plt.imshow(image)

# Plot the result
f, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 8))
f.tight_layout()

ax1.imshow(gradx, cmap='gray')
ax1.set_title('X-Direction', fontsize=20)
ax2.imshow(grady, cmap='gray')
ax2.set_title('Y-Direction', fontsize=20)
ax3.imshow(mag_binary, cmap='gray')
ax3.set_title('Gradient Magnitude', fontsize=20)
ax4.imshow(dir_binary, cmap='gray')
ax4.set_title('Gradient Direction', fontsize=20)
ax5.imshow(combined, cmap='gray')
ax5.set_title('Combination', fontsize=20)
ax6.imshow(image)
ax6.set_title('Original Image', fontsize=20)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

print("END")
"""
