'''
Input 
-> Distanztransformationen
    -> Non-maximum surpession in 5x5 Pixel > Smin of local minima
    -> sMax local Maxima from background


'''

# imports
import os
import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

dist_transforms_train_folder = "C:/Users/Tobias/Desktop/test2/train"
dist_transform_test_folder = "C:/Users/Tobias/Desktop/test2/test1"
dist_transform_test2_folder = "C:/Users/Tobias/Desktop/test2/test2"




def get_dist_transform(input_folder):
    dist_transforms = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    for dist_transform in dist_transforms:
        dist_transform_path = os.path.join(input_folder, dist_transform)
        dist_transform_array = np.load(dist_transform_path)
        get_local_Min_Max(dist_transform_array)
        print("Durchlauf Fertig")

def get_local_Min_Max(dist_transform):
    return

def watershed_segmentation(model_path):
    get_dist_transform()
    return





get_dist_transform(model_path)