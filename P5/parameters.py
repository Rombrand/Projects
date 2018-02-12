import collections
import numpy as np


# ------------------------------------------- EXTRACTION PARAMETERS ----------------------------------------------------

train = False
show = False

# Used features:
spatial_feat = True
hist_feat = True
hog_feat = True

# Params for feature extraction:
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
block_norm = 'L2-Hys'

spatial_size = (32, 32)
hist_bins = 32

# Find cars params:
ystart = 400
ystop = 600
scale = 2       # overlap 75 %

# Heat map
box_sequence = collections.deque(maxlen=25)
heat = np.zeros((720, 1280))   #).astype(np.float)
heat_threshold = 10