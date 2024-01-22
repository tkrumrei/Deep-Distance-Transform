import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import os

#######################
# calculate_average_contour_area(): calculate average area of every contour in image
def calculate_average_contour_area(image_path):
    # load image 
    image = Image.open(image_path)
    image_np = np.array(image)
    unique_pixel_values = np.unique(image_np)
    areas = []

    for pixel_value in unique_pixel_values:
        if pixel_value == 0:  # skip background
            continue

        # Binary image from pixel value
        # only this cell
        cell_binary = np.where(image_np == pixel_value, 255, 0).astype(np.uint8)

        # find conturs
        contours, _ = cv2.findContours(cell_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # calculate area for every contour and append it to the areas array
        for contour in contours:
            if len(contour) >= 3:  # polygon is only valid with min. 3 points
                polygon = Polygon(contour.squeeze())
                areas.append(polygon.area)

    # calculate average area of array area
    average_area = sum(areas) / len(areas) if areas else 0
    return average_area

#######################
# calculate_average_area_in_folder(): go through every image in folder and return the average
#                                     area of every image in the folder
def calculate_average_area_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    # filter only images
    image_files = [file for file in file_list if file.lower().endswith(('.png'))]

    total_area = 0
    count = 0

    # go through every image in folder and calculate the average contour area of it
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        average_area = calculate_average_contour_area(image_path)
        # sum up total area and increase count by 1
        total_area += average_area
        count += 1

    # calculate the total average
    overall_average = total_area / count if count > 0 else 0
    return overall_average

##### Run average area for every folder #####
# Cellpose
folder_path_Cellpose_test = "D:/Datasets/Cellpose/test/masks"  
overall_average_area_Cellpose_test = calculate_average_area_in_folder(folder_path_Cellpose_test)
print(f"Durchschnittliche Fläche Cellpose Test: {overall_average_area_Cellpose_test}")
folder_path_Cellpose_train = "D:/Datasets/Cellpose/train/masks"
overall_average_area_Cellpose_train = calculate_average_area_in_folder(folder_path_Cellpose_train)
print(f"Durchschnittliche Fläche Cellpose Train: {overall_average_area_Cellpose_train}")
print(f"Gesamtdurchschnittliche Fläche Cellpose: {(overall_average_area_Cellpose_test + overall_average_area_Cellpose_train) / 2}")
# Testis
folder_path_testis = "D:/Datasets/testis_nuclei_segmentations/coloured_masks_single_cell_nuclei"
overall_average_area_testis = calculate_average_area_in_folder(folder_path_testis)
print(f"Durchschnittliche Fläche testis: {overall_average_area_testis}")
# Total Average
print(f"Gesamtdurschnittliche Fläche Cellpose und testis: {(overall_average_area_Cellpose_test + overall_average_area_Cellpose_train + overall_average_area_testis) / 3}")

