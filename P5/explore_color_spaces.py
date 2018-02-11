import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


# Read a color image
img = cv2.imread("../../Project_Data/P5_data/small/extra00.jpeg")


# Select a small fraction of pixels to plot by subsampling it
scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

# Convert subsampled image to desired color space(s)
img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

# Plot and show
plot3d(img_small_RGB, img_small_rgb)
plt.show()

plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
plt.show()




#----------------------------------------------------------------------------------------------------------------------
# Read in an image

images = glob.glob('../../Project_Data/P5_data/small/**/*.jpeg', recursive=True)
cars = []
notcars = []
trigger = False
counter = 0
color_space = 'RGB'


for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

for index in range(100):
    if trigger == True:
        img = cv2.imread(cars[index])
        trigger = False
    else:
        img = cv2.imread(notcars[index])
        trigger = True

    """
    img = cv2.cvtColor(img, cv2.COLOR_BRG2RGB)

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    """


    # Select a small fraction of pixels to plot by subsampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)),
                           interpolation=cv2.INTER_NEAREST)

    # Convert subsampled image to desired color space(s)
    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_RGB2LUV)
    img_small_HLS = cv2.cvtColor(img_small, cv2.COLOR_RGB2HLS)
    img_small_YUV = cv2.cvtColor(img_small, cv2.COLOR_RGB2YUV)
    img_small_YCrCb = cv2.cvtColor(img_small, cv2.COLOR_RGB2YCrCb)

    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting


    #"""
    # Plot and show
    plot3d(img_small_RGB, img_small_rgb)
    plt.show()

    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    plt.show()

    plot3d(img_small_LUV, img_small_rgb, axis_labels=list("LUV"))
    plt.show()

    plot3d(img_small_HLS, img_small_rgb, axis_labels=list("HLS"))
    plt.show()

    plot3d(img_small_YUV, img_small_rgb, axis_labels=list("YUV"))
    plt.show()

    plot3d(img_small_YCrCb, img_small_rgb, axis_labels=list("YCrCb"))
    plt.show()


    #"""
    # plot images
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_small_RGB)
    ax1.set_title('RGB', fontsize=20)
    ax2.imshow(img_small_HSV)
    ax2.set_title('HSV', fontsize=20)
    ax3.imshow(img_small_LUV)
    ax3.set_title('LUV', fontsize=20)
    ax4.imshow(img_small_HLS)
    ax4.set_title('HLS', fontsize=20)
    ax5.imshow(img_small_YUV)
    ax5.set_title('YUV', fontsize=20)
    ax6.imshow(img_small_YCrCb)
    ax6.set_title('YCrCb', fontsize=20)
    plt.suptitle('This is a somewhat long figure title', fontsize=16)
    plt.show()

    #"""
