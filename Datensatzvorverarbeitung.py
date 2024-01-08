# imports
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
import os

##### Cells Length for clipping #####
# cells are usually round so 2x mean of the distance transform is width of the cell
# median and max is calculated for more information
def length_cells(folder_path):
    len_cells = np.empty(0,) # empty array for the length of the cells


    for filename in os.listdir(folder_path): # all files in folder
        if filename.endswith(".png"): # only if it ends with .png
            image_path = os.path.join(folder_path, filename) # get image path

            # image to np.array
            image = Image.open(image_path)
            image_array = np.array(image)

            # look at unique values of the distance transform
            # length of unique values is biggest value of the distance transform in this image
            unique_values = np.unique(image_array)  
            len_cells = np.append(len_cells, len(unique_values)) # append to length of cells

    mean_cell = np.mean(len_cells) # mean of the biggest values of the distance transform from the images in the folder
    median_cell = np.median(len_cells) # median of the biggest values of the distance transform from the images in the folder
    max_cell = np.max(len_cells) # biggest value of the distance transform from the images in the folder

    return(mean_cell, median_cell, max_cell)

'''
# function call length_cells:
c_test = np.array(length_cells("D:/Datasets/Cellpose/test/distance_transform/"))
print(c_test) # [24.38235294 24.         50.        ]
c_train = np.array(length_cells("D:/Datasets/Cellpose/train/distance_transform/"))
print(c_train) # [24.35740741 22.         74.        ]
t_cell = np.array(length_cells("D:/Datasets/testis_nuclei_segmentations/distance_transform"))
print(t_cell) # [13.64582539 13.         32.        ]
'''

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