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

folder_path_testis = "D:/Datasets/testis_nuclei_segmentations/img"
folder_path_cellpose_test = "D:/Datasets/Cellpose/test/img"
folder_path_cellpose_train = "D:/Datasets/Cellpose/train/img"

#######################
# get_masks(): get back mask array of image files
def get_masks(image_files):
    # mask_path = f"{image_path[:-4]}masks"
    masks_set = []
    for image in image_files:
      if len(image) < 20:
        mask_file = f"{image[:-7]}masks.png"
      else:
          mask_file = image
      masks_set.append(mask_file)
    return masks_set

#######################
# datasplits(): split files into 5 parts first one is the test set end get written into 
#                     excel file on one sheet. Remaining 4 are put together  and split again
#                     in five parts as training sets and also get written in the excel file
#                     Two other excel files will be created. One for the Cellpose dataset and
#                     another one fot the mix of the two datasets.
def datasplits(folder_path, folder_path_cellpose_test, folder_path_cellpose_train):
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

    ##### Excel files #####
    # Testis:
    # create excel file and add names of images from the cellpose and testis test sets
    # on one sheet. Also create five other sheets for every fold of testis training sets
    excel_file_path = os.path.join(folder_path, 'testis_split.xlsx')
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        # training sets
        for i, set in enumerate(training_sets):
            df_train = pd.DataFrame({
                'Training Image Files': set,
                'Training Mask Files': get_masks(set)})
            df_train.to_excel(writer, sheet_name=f'Fold {i + 1}', index=False)
        # test sets
        df_test = pd.DataFrame({
            'Test Image Files': image_files_cellpose_test,
            'Test Mask Files': get_masks(image_files_cellpose_test)})
        df_test.to_excel(writer, sheet_name='Cellpose Test Set', index=False)
        df_test = pd.DataFrame({
            'Test Image Files': test_set,
            'Test Mask Files': get_masks(test_set)})
        df_test.to_excel(writer, sheet_name='Testis Test Set', index=False)

    # Cellpose:
    # create excel file and add names of images from the cellpose and testis test sets
    # on one sheet. Create also another sheet for training with the cellpose train dataset 
    excel_file_path = os.path.join(folder_path, 'cellpose_split.xlsx')
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        # training set
        df_train = pd.DataFrame({
            'Training Image Files': image_files_cellpose_train,
            'Training Mask Files': get_masks(image_files_cellpose_train)})
        df_train.to_excel(writer, sheet_name='Training Set', index=False)
        # test sets
        df_test = pd.DataFrame({
            'Test Image Files': image_files_cellpose_test,
            'Test Mask Files': get_masks(image_files_cellpose_test)})
        df_test.to_excel(writer, sheet_name='Cellpose Test Set', index=False)
        df_test = pd.DataFrame({
            'Test Image Files': test_set,
            'Test Mask Files': get_masks(test_set)})
        df_test.to_excel(writer, sheet_name='Testis Test Set', index=False)
            
    # Mix:
    # create excel file and add names of images from the cellpose and testis test sets
    # on one sheet. Also create five other sheets for every fold of testis training sets 
    # combined with the Cellpose training set
    excel_file_path = os.path.join(folder_path, 'mix_split.xlsx')
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        # training sets
        for i, set in enumerate(training_sets_mix):
            df_train = pd.DataFrame({
                'Training Image Files': set,
                'Training Mask Files': get_masks(set)})
            df_train.to_excel(writer, sheet_name=f'Fold {i + 1}', index=False)
        # test sets
        df_test = pd.DataFrame({
            'Test Image Files': image_files_cellpose_test,
            'Test Mask Files': get_masks(image_files_cellpose_test)})
        df_test.to_excel(writer, sheet_name='Cellpose Test Set', index=False)
        df_test = pd.DataFrame({
            'Test Image Files': test_set,
            'Test Mask Files': get_masks(test_set)})
        df_test.to_excel(writer, sheet_name='Testis Test Set', index=False)

datasplits(folder_path_testis, folder_path_cellpose_test, folder_path_cellpose_train)