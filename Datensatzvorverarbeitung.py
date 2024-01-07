# imports
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import os

########## Distanztransformation ##########
# Cellpose

##### Train dataset #####
'''
folder_path = "D:/Datasets/Cellpose/train/masks/"
folder_output = "D:/Datasets/Cellpose/train/DistanceTransform/"

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)

        # Bild für Verarbeitung bereit machen
        image = Image.open(image_path)
        image_array = np.array(image)
        unique_values = np.unique(image_array)
        combined_dist_transform = np.zeros_like(image_array, dtype=float) # leeres np Array für Distanztransformation erstellen

        # Distanztransformationen
        # Distanztransformation für jeden einzigartigen Pixelwert berechnen
        for value in unique_values[1:]:  # Ab 1 beginnen
            # Binäres Bild, in dem nur aktuelle Pixelwertgruppe Wert 1 hat, Rest Wert 0
            binary_mask = (image_array == value).astype(np.uint8)
            
            # Distanztransformation berechnen
            dist_transform = distance_transform_edt(binary_mask)
            
            # Distanzwerte normalisieren falls größer als 255
            normalized_dist_transform = (dist_transform * 255) / dist_transform.max()
            #!!! (dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255
            
            # berechnete Distanztransformation von value in np Array hinzufügen
            combined_dist_transform += dist_transform
        
        prefix_filename = filename[:3]
        saved_image_path = os.path.join(folder_output, f"{prefix_filename}_DT.png")
        saved_array_path = os.path.join(folder_output, f"{prefix_filename}_DT.npy")
        # In Bild umwandeln und Array und Bild abspeichern
        combined_dist_transform_image = Image.fromarray(combined_dist_transform.astype(np.uint8))
        # combined_dist_transform_image.show()
        combined_dist_transform_image.save(saved_image_path)
        np.save(saved_array_path, combined_dist_transform)
'''

##### Test dataset #####
'''
folder_path = "D:/Datasets/Cellpose/test/masks/"
folder_output = "D:/Datasets/Cellpose/test/DistanceTransform/"

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)

        # Bild für Verarbeitung bereit machen
        image = Image.open(image_path)
        image_array = np.array(image)
        unique_values = np.unique(image_array)
        combined_dist_transform = np.zeros_like(image_array, dtype=float) # leeres np Array für Distanztransformation erstellen

        # Distanztransformationen
        # Distanztransformation für jeden einzigartigen Pixelwert berechnen
        for value in unique_values[1:]:  # Ab 1 beginnen
            # Binäres Bild, in dem nur aktuelle Pixelwertgruppe Wert 1 hat, Rest Wert 0
            binary_mask = (image_array == value).astype(np.uint8)
            
            # Distanztransformation berechnen
            dist_transform = distance_transform_edt(binary_mask)
            
            # Distanzwerte normalisieren falls größer als 255
            normalized_dist_transform = (dist_transform * 255) / dist_transform.max()
            #!!! (dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255
            
            # berechnete Distanztransformation von value in np Array hinzufügen
            combined_dist_transform += dist_transform
        
        prefix_filename = filename[:3]
        saved_image_path = os.path.join(folder_output, f"{prefix_filename}_DT.png")
        saved_array_path = os.path.join(folder_output, f"{prefix_filename}_DT.npy")
        # In Bild umwandeln und Array und Bild abspeichern
        combined_dist_transform_image = Image.fromarray(combined_dist_transform.astype(np.uint8))
        # combined_dist_transform_image.show()
        combined_dist_transform_image.save(saved_image_path)
        np.save(saved_array_path, combined_dist_transform)
'''

# Testis 

##### masks_single_cell_nuclei dataset #####
'''
folder_path = "D:/Datasets/testis_nuclei_segmentations/masks_single_cell_nuclei"
folder_output = "D:/Datasets/testis_nuclei_segmentations/CellDistanceTransform/"

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)

        # Bild für Verarbeitung bereit machen
        image = Image.open(image_path)
        image_array = np.array(image)
        unique_values = np.unique(image_array)
        combined_dist_transform = np.zeros_like(image_array, dtype=float) # leeres np Array für Distanztransformation erstellen

        # Distanztransformationen
        # Distanztransformation für jeden einzigartigen Pixelwert berechnen
        for value in unique_values[1:]:  # Ab 1 beginnen
            # Binäres Bild, in dem nur aktuelle Pixelwertgruppe Wert 1 hat, Rest Wert 0
            binary_mask = (image_array == value).astype(np.uint8)
            
            # Distanztransformation berechnen
            dist_transform = distance_transform_edt(binary_mask)
            
            # Distanzwerte normalisieren falls größer als 255
            normalized_dist_transform = (dist_transform * 255) / dist_transform.max()
            #!!! (dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255
            
            # berechnete Distanztransformation von value in np Array hinzufügen
            combined_dist_transform += dist_transform

        only_filename = filename[:-4]

        saved_image_path = os.path.join(folder_output, f"{only_filename}_DT.png")
        saved_array_path = os.path.join(folder_output, f"{only_filename}_DT.npy")
        # In Bild umwandeln und Array und Bild abspeichern
        combined_dist_transform_image = Image.fromarray(combined_dist_transform.astype(np.uint8))
        combined_dist_transform_image.save(saved_image_path)
        np.save(saved_array_path, combined_dist_transform)
'''
        
##### masks_tubulus dataset #####
'''
folder_path = "D:/Datasets/testis_nuclei_segmentations/masks_tubulus"
folder_output = "D:/Datasets/testis_nuclei_segmentations/TubulusDistanceTransform/"

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)

        # Bild für Verarbeitung bereit machen
        image = Image.open(image_path)
        image_array = np.array(image)
        unique_values = np.unique(image_array)
        combined_dist_transform = np.zeros_like(image_array, dtype=float) # leeres np Array für Distanztransformation erstellen

        # Distanztransformationen
        # Distanztransformation für jeden einzigartigen Pixelwert berechnen
        for value in unique_values[1:]:  # Ab 1 beginnen
            # Binäres Bild, in dem nur aktuelle Pixelwertgruppe Wert 1 hat, Rest Wert 0
            binary_mask = (image_array == value).astype(np.uint8)
            
            # Distanztransformation berechnen
            dist_transform = distance_transform_edt(binary_mask)
            
            # Distanzwerte normalisieren falls größer als 255
            normalized_dist_transform = (dist_transform * 255) / dist_transform.max()
            #!!! (dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255
            
            # berechnete Distanztransformation von value in np Array hinzufügen
            combined_dist_transform += dist_transform
        

        only_filename = filename[:-4]

        saved_image_path = os.path.join(folder_output, f"{only_filename}_DT.png")
        saved_array_path = os.path.join(folder_output, f"{only_filename}_DT.npy")
        # In Bild umwandeln und Array und Bild abspeichern
        combined_dist_transform_image = Image.fromarray(combined_dist_transform.astype(np.uint8))
        combined_dist_transform_image.save(saved_image_path)
        np.save(saved_array_path, combined_dist_transform)
'''

########## Umbenennung Testis Datensätze ##########
'''
def rename_files_in_folder(folder_path, type):
    count = 1 # Zähler
    num_zeros = 4 # 0en

    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):

            num_with_zeros = str(count).zfill(num_zeros) # führende 0en hinzufügen
            new_filename = f"{num_with_zeros}_{type}.png"

            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)

            os.rename(old_filepath, new_filepath)
            count += 1

# Funktionsaufruf
rename_files_in_folder("D:/Datasets/testis_nuclei_segmentations/original_without_background/", "img")
'''

########## Zusammenführen der Datensätze ##########

# Img
##### Cellpose train dataset #####

##### #####

# Distance Transform
##### Cellpose train dataset #####        
