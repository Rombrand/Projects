import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from params import *  # Left, Right


#image = cv2.imread('test_images/warp_img_with_lines.jpg')  #straight_lines1  test5
test_image = cv2.imread('test_images/test5.jpg') #straight_lines1  test5

# Convert to RGB
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)


def colour_extraction(image, r_thresh =  ([220, 255]), s_thresh = ([120, 255])):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)


    H = hls[:,:,0]
    L = hls[:,:,1]  # For shadows
    S = hls[:,:,2]  # general good detection

    R = image[:,:,0]
    G = image[:,:,1]  # For shadows
    B = image[:,:,2]

    #plt.imshow(G)
    #plt.show()


    #---------------------------- R-CHANNEL-FILTER -----------------------------
    retval, r_binary = cv2.threshold(R.astype('uint8'), r_thresh[0], r_thresh[1], cv2.THRESH_BINARY)
    #r_binary = np.zeros_like(R)
    #r_binary[(R >= thresh[0]) & (R <= thresh[1])] = 1
    #---------------------------- S-CHANNEL-FILTER -----------------------------


    retval, s_binary = cv2.threshold(S.astype('uint8'), s_thresh[0], s_thresh[1], cv2.THRESH_BINARY)
    #s_binary = np.zeros_like(R)
    #s_binary[(S >= thresh[0]) & (S <= thresh[1])] = 1

    #---------------------------- COMBINING CHANNELS ---------------------------

    # Combine the two images to a binary thresholds
    combined_binaries_255 = cv2.bitwise_or(r_binary, s_binary)
    combined_binaries = np.zeros_like(combined_binaries_255)
    combined_binaries[(combined_binaries_255 > 0)] = 1

    if show == True:
        # Plot results line extraction
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
        f.tight_layout()
        ax1.imshow(s_binary,  cmap='gray')
        ax1.set_title('R-Binary', fontsize=20)
        ax2.imshow(r_binary, cmap='gray')
        ax2.set_title('S-Binary', fontsize=20)
        ax3.imshow(combined_binaries, cmap='gray')
        ax3.set_title('Combined', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return combined_binaries




#--------------------------- GRADIENT -----------------------------------------
def gradient_extraction(img, gauss_kernel = 15, sobel_kernel=3, orient = 'both', thresh=(20, 255)):
    # Apply the following steps to img
    #  Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x, y direction separately, or in both directions. Transform the result to an absolute value
    cv2.GaussianBlur(gray, (gauss_kernel, gauss_kernel), 0)


    if orient == 'x':
        sob = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sob = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        sob = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sob_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # calculate the direction of the gradient
    abs_sob = np.abs(sob)
    #abs_sob_y = np.abs(sob_y)
    #gradient_direction = np.arctan2(sob_abs_y, sob_abs_x)
    scaled_sob= np.uint8(255*abs_sob/np.max(abs_sob))
    #scaled_sob = cv2.convertScaleAbs(sob, alpha=1, beta=0)

    # Create a binary mask
    binary = np.zeros_like(scaled_sob)
    binary[(scaled_sob >= thresh[0]) & (scaled_sob <= thresh[1])] = 1
    return binary

"""

#plt.figure(figsize=(30, 8))
plt.subplots(2, 4, figsize = (20, 7))
for i in range(1,9):
    thresh = ([30, 255])
    thresh[0] = 5 + 2 * i
    binary = gradient_extraction(test_image, sobel_kernel=3, orient='x', thresh=thresh)
    plt.subplot(2, 4, i)
    if i == 1:
        plt.imshow(test_image)
    else:
        plt.imshow(binary, cmap='gray')
    plt.title('Sobel, Lower Thresh = {}'.format(thresh[0]))
    plt.axis('off')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

"""