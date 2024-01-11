# imports
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
import os

##### Signed Distance Transform #####


#######################
# scale_to_255(): scales array from 0 to 255 and gives image for visual control
def scale_to_255(array_to_scale):
    min_value = np.min(array_to_scale)
    max_value = np.max(array_to_scale)

    # scale and round array
    scaled_array = 255 * (array_to_scale - min_value) / (max_value - min_value)
    scaled_array = np.round(scaled_array).astype(np.uint8)

    # show image of array
    scaled_image = Image.fromarray((scaled_array).astype(np.uint8))
    scaled_image.show()
     


#######################
# sign(): gets distance transform of the masks and distance transform of the background
#         and brings them together as one. The distance transform of the masks is
#         multiplied by -1 and copied to the signed distance transform.
def sign(dist_transform, bg_dist_transform):
    signed = bg_dist_transform

    signed[dist_transform > 0] = -1 * dist_transform[dist_transform > 0]
    
    return signed        
     
#######################
# scale(): gets the signed distance transform and clip it to +-25 and scale it from
#          -0.9 to 0.9.
def scale(signed):
    # clip
    clipped = np.clip(signed, -25, 25)
    min_value = np.min(clipped)
    max_value = np.max(clipped)
    
    # scale 
    scaled = -0.9 + (clipped - min_value) * (1.8 / (max_value - min_value))
    
    return scaled 

#######################
# weight_function(): get's the binary array mask and calculates the weights for it
def weight_function(binary_array):
    sigma = 3.0  
    weights = gaussian_filter(binary_array.astype(float), sigma)

    weights /= np.percentile(weights, 99) # normalise the weighted function
    
    return weights

#######################
# signed_distance_transform(): calculates signed distance transformation and weights
def signed_distance_transform(input_folder, output_folder_dt, output_folder_w):
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)

        # image to np.array
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # array to binary array
        binary_array = np.where(image_array > 0, 1, 0)
        bg_binary_array = np.where(image_array == 0, 1, 0)

        # calculate weights from the binary array map
        weights = weight_function(binary_array)
        # only for visual control
        #print(weights)

        ### signed distance transform ###
        # distance transformations for masks and background
        dist_transform = distance_transform_edt(binary_array)
        bg_dist_transform = distance_transform_edt(bg_binary_array)
        # only for visual control
        #scale_to_255(dist_transform)
        #scale_to_255(bg_dist_transform)
        
        # calculate signed distance transformation 
        signed = sign(dist_transform, bg_dist_transform)
        # only for visual control
        #scale_to_255(signed)
        
        # calculate scaled distance transformation
        scaled = scale(signed)
        # only for visual control
        #scale_to_255(scaled)

        # save files
        if len(filename) < 20:
            prefix_filename = filename[:3]
            saved_array_path = os.path.join(output_folder_dt, f"0{prefix_filename}_dt")
            np.save(saved_array_path, scaled)

            saved_array_path = os.path.join(output_folder_w, f"0{prefix_filename}_w")
            np.save(saved_array_path, weights)
        else:
            saved_array_path = os.path.join(output_folder_dt, filename)
            np.save(saved_array_path, scaled)

            saved_array_path = os.path.join(output_folder_w, filename)
            np.save(saved_array_path, weights)

# function call signed_distance_transform
signed_distance_transform("D:/Datasets/Cellpose/test/masks/", "D:/Datasets/Cellpose/test/distance_transform/", "D:/Datasets/Cellpose/test/weights/")
signed_distance_transform("D:/Datasets/Cellpose/train/masks/", "D:/Datasets/Cellpose/train/distance_transform/", "D:/Datasets/Cellpose/train/weights/")
signed_distance_transform("D:/Datasets/testis_nuclei_segmentations/masks_tubulus/", "D:/Datasets/testis_nuclei_segmentations/distance_transform_tubulus/", "D:/Datasets/testis_nuclei_segmentations/weights_tubulus/")
signed_distance_transform("D:/Datasets/testis_nuclei_segmentations/coloured_masks_single_cell_nuclei/", "D:/Datasets/testis_nuclei_segmentations/distance_transform_cell/", "D:/Datasets/testis_nuclei_segmentations/weights_cell/")
'''
file = "D:/Datasets/Cellpose/test/distance_transform/0000_dt.npy"
loaded = scale_to_255(np.load(file))
control = Image.fromarray(loaded.astype(np.uint8))
control.show()
'''