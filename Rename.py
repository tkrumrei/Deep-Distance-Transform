''' 
Rename files and move them to another folder
'''
# imports
import os

folder_path = 'C:/Users/Tobias/Desktop/test/alt'
new_folder_path = 'D:/Datasets/Joined'

'''
# Überprüfen, ob der Ordner existiert
if not os.path.exists(folder_path):
    print("Ordner nicht gefunden")
else:
    # Schleife durch alle Dateien im Ordner
    for file_name in os.listdir(folder_path):
        # Vollständiger Pfad der aktuellen Datei
        full_file_path = os.path.join(folder_path, file_name)

        # Überprüfen, ob es sich um eine Datei handelt und nicht um einen Ordner
        if os.path.isfile(full_file_path):
            # Trennen des Dateinamens bei ".png" und nehmen des ersten Teils
            #new_file_name = file_name.split('_')[0]
            new_file_name = f"0{file_name}"

            # Hinzufügen der restlichen Dateiendung, falls vorhanden
            #new_file_name += file_name.split('.png')[-1]

            # Vollständiger Pfad der neuen Datei
            new_file_path = os.path.join(folder_path, new_file_name)

            # Umbenennen der Datei
            os.rename(full_file_path, new_file_path)

    print("Dateien wurden umbenannt.")
'''

#######################
# rename_starting_numbers(): rename the file and change starting number
def rename_starting_numbers(starting_number, folder_path):

    # sort files for safety 
    files = sorted(os.listdir(folder_path))

    # go through all folders
    for index, file_name in enumerate(files):
        # preparations to rename the file
        full_file_path = os.path.join(folder_path, file_name)
        file_name = file_name.split('_')[1] # split until after the "_"
        print(file_name)
        # leading zero if necessary
        if starting_number  < 1000:
            new_file_name = f"0{starting_number}_{file_name}"
            starting_number += 1
        else:
            new_file_name = f"{starting_number}_{file_name}"
            starting_number += 1

        # make new file path
        new_file_path = os.path.join(folder_path, new_file_name)

        # rename file
        os.rename(full_file_path, new_file_path)
        
    print("Files starting number were renamed")

'''
# Rename Cellpose test to starting number 540
rename_starting_numbers(540, "D:/Datasets/Cellpose/test/distance_transform")
rename_starting_numbers(540, "D:/Datasets/Cellpose/test/img")
rename_starting_numbers(540, "D:/Datasets/Cellpose/test/masks")
rename_starting_numbers(540, "D:/Datasets/Cellpose/test/weights")
'''

#######################
# rename_for_Cellpose(): files will be renamed with the abbrevation in name
def rename_for_Cellpose(abbrevation, folder_path):
    # sort files for safety 
    files = sorted(os.listdir(folder_path))

    # go through all folders
    for index, file_name in enumerate(files):
        # preparations to rename the file
        full_file_path = os.path.join(folder_path, file_name)
        file = file_name.split('.', 1)[1] # split after the "."
        file_name = file_name.split('.', 1)[0] # split before the "."
        
        # add abbrevation to file name
        new_file_name = f"{file_name}{abbrevation}.{file}"

        # make new file path
        new_file_path = os.path.join(folder_path, new_file_name)

        # rename file
        os.rename(full_file_path, new_file_path)
        
    print(f"Files were renamed to: ...{abbrevation}...")

'''
rename_for_Cellpose("_img", "D:/Datasets/testis_nuclei_segmentations/img") # img
rename_for_Cellpose("_masks", "D:/Datasets/testis_nuclei_segmentations/masks") # masks
'''