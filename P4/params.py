
import numpy as np
import cv2
import pickle


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        #average x values of the fitted line over the last n iterations
        self.bestx = None

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

      #  #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

      # #radius of curvature of the line in some units
        self.radius_of_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

      #  #x values for detected line pixels
        self.allx = None

      #  #y values for detected line pixels
        self.ally = None



# Right and Left lane line
Left = Line()
Right = Line()


# ---------------- Params for Camera_Calibration Module ---------------
nx = 9  # the number of inside corners in x
ny = 6  # the number of inside corners in y

test_image = cv2.imread('test_images/straight_lines1.jpg') #straight_lines1  test5

img_size = test_image.shape[1::-1]
size_x = img_size[0]
size_y = img_size[1]



# Define Source and Destination points
x1_offset = 200
x2_offset = 595     #570
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

src = np.float32([P1_src, P2_src, P3_src, P4_src])
dst = np.float32([P1_dst, P2_dst, P3_dst, P4_dst])




# ---------------- Params for line_detection Module ---------------

r_thresh = ([220, 255])
s_thresh = ([120, 255])
thresh_gradient = (20, 255)

gauss_kernel = 15
sobel_kernel = 3
morph_kernel = np.ones((2, 2), np.uint8)

# ------------------ Params for calculations -----------------------
ploty = [np.array([False])]
left_fitx = []
right_fitx = []

show = False