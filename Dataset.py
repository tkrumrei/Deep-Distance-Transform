'''
'''
# imports
import numpy as np

# Read image and mask

# convert to grayscale array
# crop to 512 x 512 
#   > make sure at least 25% are covered by cells with masks

# split into train, test and validate
# with double 5 fold cross validation
#   > split 5 sets --> one test set
#       > joined remaining 4 sets joined and also split into 5 sets --> one validation set
# saved into the folders

# data augmentation for training sets
# mirrored version
# rotate image several times 90, 180, 270 degrees