import opencv
import numpy as np
from PIL import Image

def calculate_cell_sizes(image):
    # preperation
    image_np = np.array(image)
    unique_pixel_values = np.unique(image_np)
    cell_sizes = []

    for pixel_value in unique_pixel_values:
        if pixel_value == 0: # not background
            continue

        # binary image for pixel value
        cell_binary = np.zeros_like(image_np, dtype=np.uint8)
        cell_binary[image_np == pixel_value] = 255

        # find contours, calculate the area of it and append to list
        contours, _ = opencv.findContours(cell_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = opencv.contourArea(contours[0]) if contours else 0
        cell_sizes.append(area)

    # calculate average cell size
    average_cell_size = sum(cell_sizes) / len(cell_sizes)
    return average_cell_size


image = Image.open("D:/Datasets/Cellpose/test/masks/040_masks.png")
print(calculate_cell_sizes(image))