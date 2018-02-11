

# ------------------------------------------- EXTRACTION PARAMETERS ----------------------------------------------------

train = False

# Used features:
spatial_feat = True
hist_feat = True
hog_feat = True

# Params for feature extraction:
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
block_norm = 'L2-Hys'

spatial_size = (32, 32)
hist_bins = 64

# Find cars params:
ystart = 400
ystop = 600
scale = 2       # overlap 75 %