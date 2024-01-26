import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import os

#######################
# calculate_average_contours(): calculate average area of every contour in image
def calculate_average_contours(image_path):
    # load image 
    image = Image.open(image_path)
    image_np = np.array(image)
    unique_pixel_values = np.unique(image_np)
    areas = []
    widths = []
    heights = []

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
                min_x, min_y, max_x, max_y = polygon.bounds
                areas.append(polygon.area)
                widths.append(max_x - min_x)
                heights.append(max_y - min_y)



    # calculate average area of array area
    average_area = sum(areas) / len(areas) 
    average_width = sum(widths) / len(widths)
    average_height = sum(heights) / len(heights)

    return average_area, average_width, average_height

#######################
# calculate_average_area_in_folder(): go through every image in folder and return the average
#                                     area of every image in the folder
def calculate_average_area_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    # filter only images
    image_files = [file for file in file_list if file.lower().endswith(('.png'))]

    total_area = 0
    total_width = 0
    total_height = 0
    count = 0

    # go through every image in folder and calculate the average contour area of it
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        average_area, average_width, average_height = calculate_average_contours(image_path)
        # sum up total area and increase count by 1
        total_area += average_area
        total_width += average_width
        total_height += average_height
        count += 1

    # calculate the total average
    overall_average_area = total_area / count
    overall_average_width = total_width / count
    overall_average_height = total_height / count

    return overall_average_area, overall_average_width, overall_average_height

##### Run average area for every folder #####
# Cellpose
folder_path_Cellpose_test = "D:/Datasets/Cellpose/test/masks"  
overall_average_area_Cellpose_test, overall_average_width_Cellpose_test, overall_average_height_Cellpose_test = calculate_average_area_in_folder(folder_path_Cellpose_test)
print(f"Cellpose Test: Durchschnittliche Fläche = {overall_average_area_Cellpose_test}, Breite = {overall_average_width_Cellpose_test}, Höhe = {overall_average_height_Cellpose_test}")
folder_path_Cellpose_train = "D:/Datasets/Cellpose/train/masks"
overall_average_area_Cellpose_train, overall_average_width_Cellpose_train, overall_average_height_Cellpose_train = calculate_average_area_in_folder(folder_path_Cellpose_train)
print(f"Cellpose Train: Durchschnittliche Fläche = {overall_average_area_Cellpose_train}, Breite = {overall_average_width_Cellpose_train}, Höhe = {overall_average_height_Cellpose_train}")
print(f"Gesamtdurchschnittliche Fläche Cellpose: {(overall_average_area_Cellpose_test + overall_average_area_Cellpose_train) / 2}")
print(f"Gesamtdurchschnittliche Breite Cellpose: {(overall_average_width_Cellpose_test + overall_average_width_Cellpose_test) / 2}")
print(f"Gesamtdurchschnittliche Höhe Cellpose: {(overall_average_height_Cellpose_test + overall_average_height_Cellpose_test) / 2}")
# Testis
folder_path_testis = "D:/Datasets/testis_nuclei_segmentations/masks"
overall_average_area_testis, overall_average_width_testis, overall_average_height_testis = calculate_average_area_in_folder(folder_path_testis)
print(f"Testis: Durchschnittliche Fläche = {overall_average_area_testis}, Breite = {overall_average_width_testis}, Höhe = {overall_average_height_testis}")
# Total Average
print(f"Gesamtdurschnittliche Fläche Cellpose und testis: {(overall_average_area_Cellpose_test + overall_average_area_Cellpose_train + overall_average_area_testis) / 3}")
print(f"Gesamtdurschnittliche Breite Cellpose und testis: {(overall_average_width_Cellpose_test + overall_average_width_Cellpose_train + overall_average_width_testis) / 3}")
print(f"Gesamtdurschnittliche Höhe Cellpose und testis: {(overall_average_height_Cellpose_test + overall_average_height_Cellpose_train + overall_average_height_testis) / 3}")