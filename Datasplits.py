'''
Testsets:
- I need a test set of the testis dataset because the Cellpose dataset already has one
  - split testis set in 5 sets --> first one is test set

- Cellpose training data don't get a 5-fold cross-validation because...
  - the different groups of sources are not known therefor I don't know which image is in 
    which group. So I cannot make sure that every fold has the same procentual amount of every
    group.
  - it would be to time consuming to figure it out by myself-

- Testis 5-fold cross validation:
  - put remaining 4 testis sets together and split them again in 5 sets to get 5 folds

- Mixed 5-fold cross validation:
  - combine every fold of the testis 5-fold cross-validation with the Cellpose training set
'''

# imports
import os
import pandas as pd
from math import ceil
import random
import shutil

folder_path_testis = "D:/Datasets/testis_nuclei_segmentations/img"
folder_path_cellpose_test = "D:/Datasets/Cellpose/test/img"
folder_path_cellpose_train = "D:/Datasets/Cellpose/train/img"
output_folder_csv = "D:/Datasets"

#######################
# get_masks(): get back mask array of image files
def get_masks(image_files):
    masks_set = []
    for image in image_files:
       mask_file = f"{image[:-7]}masks.png"
       masks_set.append(mask_file)
    return masks_set


#######################
# get_distance_transform(): get back distance transform array of image files
def get_distance_transform(image_files):
    dt_set = []
    for image in image_files:
       dt_file = f"{image[:-7]}dt.npy"
       dt_set.append(dt_file)
    return dt_set


#######################
# get_weights(): get back weights array of image files
def get_weights(image_files):
    weights_set = []
    for image in image_files:
       weights_file = f"{image[:-7]}weights.npy"
       weights_set.append(weights_file)
    return weights_set

#######################
# get_all_to_csv(): gets an array of image files and adds them and the masks, dt and weights
#                   to .csv files
def get_all_to_csv(array, name, output_folder):
    df = pd.DataFrame(array)
    df.to_csv(f"{output_folder}/{name}_img.csv")
    df = pd.DataFrame(get_masks(array))
    df.to_csv(f"{output_folder}/{name}_masks.csv")
    df = pd.DataFrame(get_distance_transform(array))
    df.to_csv(f"{output_folder}/{name}_dt.csv")
    df = pd.DataFrame(get_weights(array))
    df.to_csv(f"{output_folder}/{name}_weights.csv")

#######################
# datasplits(): makes the datasplit for the 5-fold cross-validation
def datasplits(folder_path, folder_path_cellpose_test, folder_path_cellpose_train, output_folder):
    # get Cellpose datasets
    image_files_cellpose_test = [f for f in os.listdir(folder_path_cellpose_test) if f.endswith('.png')]
    image_files_cellpose_train = [f for f in os.listdir(folder_path_cellpose_train) if f.endswith('.png')]

    # make list of all images in folder path and randomize order
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    random.shuffle(image_files)

    # get size of images and calculate size for every set
    files_len = len(image_files)
    set_size = ceil(files_len / 5) # round up

    # First part is test set
    test_set = image_files[0:set_size]

    # prepare rest of the files for training sets array and create training_sets_mix array
    training_set = image_files[set_size:]
    training_set_size = ceil(len(training_set) / 5) # round up 
    training_sets = [] 
    training_sets_mix = []

    # make array for every set and add it to training sets array
    for start in range(0, len(training_set), training_set_size):
        end = start + training_set_size
        set_slice = training_set[start:end]
        training_sets.append(set_slice)
        # combine every set with cellpose train dataset
        training_sets_mix.append(set_slice + image_files_cellpose_train)

    ##### CSV files #####
    # Cellpose
    get_all_to_csv(image_files_cellpose_train, "Cellpose_train", output_folder)
    get_all_to_csv(image_files_cellpose_test, "Cellpose_test", output_folder)
    # Testis
    for i, set in enumerate(training_sets):
            get_all_to_csv(set, f"Testis_fold_{i + 1}", output_folder)
    get_all_to_csv(test_set, "Testis_test", output_folder)
    # Mix
    for i, set in enumerate(training_sets_mix):
        get_all_to_csv(set, f"Mix_fold_{i + 1}", output_folder)

    # return
    return test_set, training_sets    


#######################
# split_Testis_data(): splits images and masks of testis from the test and folds in different folders
def split_Testis_data():
    testis_test, testis_train = datasplits(folder_path_testis, folder_path_cellpose_test, folder_path_cellpose_train, output_folder_csv)

    input_folder = "D:/Datasets/Cellpose_Model/Testis/img"
    output_folder = "D:/Datasets/Cellpose_Model/Testis"
    input_folder_masks = "D:/Datasets/Cellpose_Model/Testis/masks"

    test_masks = get_masks(testis_test)

    # Testis Test Set image
    for image in testis_test:
        old_image_path = os.path.join(input_folder, image)
        new_output_folder = f"{output_folder}/test"
        new_image_path = os.path.join(new_output_folder, image)

        shutil.move(old_image_path, new_image_path)
    # Testis Test Set mask    
    for image in test_masks:
        old_image_path = os.path.join(input_folder_masks, image)
        new_output_folder = f"{output_folder}/test_masks"
        new_image_path = os.path.join(new_output_folder, image)

        shutil.move(old_image_path, new_image_path)

    # Testis Folds
    for i, set in enumerate(testis_train):
        for image in set:
            old_image_path = os.path.join(input_folder, image)
            new_output_folder = f"{output_folder}/Fold_{i + 1}"
            new_image_path = os.path.join(new_output_folder, image)

            shutil.move(old_image_path, new_image_path)
        # Testis Test Set mask
        train_masks = get_masks(set)    
        for image in train_masks:
            old_image_path = os.path.join(input_folder_masks, image)
            new_output_folder = f"{output_folder}/Fold_{i + 1}_masks"
            new_image_path = os.path.join(new_output_folder, image)

            shutil.move(old_image_path, new_image_path)

split_Testis_data()
