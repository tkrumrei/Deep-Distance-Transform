import os
import pandas as pd
import random

# Pfad zum Ordner mit PNG-Dateien
input_folder = "D:/Datasets/Cellpose/train/img"

# List of images in folder and randomize order
png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
random.shuffle(png_files)

# calculate split size and
num_splits = 5
split_size = len(png_files) // num_splits

# Teile die Dateien in Splits auf
splits = [png_files[i * split_size:(i + 1) * split_size] for i in range(num_splits)]

# Erstelle DataFrame für die CSV-Datei
csv_data = {f'Split {i+1}': split for i, split in enumerate(splits)}

# Fülle die Splits mit NaN-Werten auf, um sicherzustellen, dass alle Splits die gleiche Länge haben
max_length = max(map(len, splits))
for key in csv_data:
    csv_data[key] += [''] * (max_length - len(csv_data[key]))

# Erstelle DataFrame und speichere CSV-Datei
df = pd.DataFrame(csv_data)
df.to_csv('file_list.csv', index=False)