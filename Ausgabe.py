from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

result_path = "D:/Bachelorarbeit/Ergebnisse"

# Functions for additional Metrics: 
def recall(array):
    return array[2] / (array[2] + array[4])

def real_precision(array):
    return array[2] / (array[2] + array[3])

def f1_score(array):
    recall_rate = recall(array)
    precision = real_precision(array)
    # should be 1 not 0
    return 2 * ((precision * recall_rate) / (precision + recall_rate))

def calculate_addi_metrics(array):
    add_metrics = [recall(array), real_precision(array), f1_score(array)]
    array = np.append(array, add_metrics)
    return array

##### Cellpose Modelzoo #####

modelzoopath = f"{result_path}/Cellpose_Modelzoo"
'''
# Nuclei Model
nuclei_cellpose_path = f"{modelzoopath}/Ohne_Alles_Evaluiert/nuclei_metrics_Cellpose.npy"
nuclei_testis_path = f"{modelzoopath}/Ohne_Alles_Evaluiert/nuclei_metrics_Testis.npy"

nuclei_cellpose = np.load(nuclei_cellpose_path)
nuclei_cellpose = calculate_addi_metrics(nuclei_cellpose)
nuclei_testis = np.load(nuclei_testis_path)
nuclei_testis = calculate_addi_metrics(nuclei_testis)

print(nuclei_cellpose)
print(nuclei_testis)

# Cyto Model
cyto_cellpose_path = f"{modelzoopath}/Ohne_Alles_Evaluiert/cyto_metrics_Cellpose.npy"
cyto_testis_path = f"{modelzoopath}/Ohne_Alles_Evaluiert/cyto_metrics_Testis.npy"

cyto_cellpose = np.load(cyto_cellpose_path)
cyto_cellpose = calculate_addi_metrics(cyto_cellpose)
cyto_testis = np.load(cyto_testis_path)
cyto_testis = calculate_addi_metrics(cyto_testis)

print(cyto_cellpose)
print(cyto_testis)
'''
# Testis 
'''
M_fold_1_T = np.load(f"{modelzoopath}/Testis/cyto_metrics_Fold_1.npy")
M_fold_1_T = calculate_addi_metrics(M_fold_1_T)
print(M_fold_1_T)

M_fold_2_T = np.load(f"{modelzoopath}/Testis/cyto_metrics_Fold_2.npy")
M_fold_2_T = calculate_addi_metrics(M_fold_2_T)
print(M_fold_2_T)
M_fold_3_T = np.load(f"{modelzoopath}/Testis/cyto_metrics_Fold_3.npy")
M_fold_3_T = calculate_addi_metrics(M_fold_3_T)
print(M_fold_3_T)

M_fold_4_T = np.load(f"{modelzoopath}/Testis/cyto_metrics_Fold_4.npy")
M_fold_4_T = calculate_addi_metrics(M_fold_4_T)
print(M_fold_4_T)
M_fold_5_T = np.load(f"{modelzoopath}/Testis/cyto_metrics_Fold_5.npy")
M_fold_5_T = calculate_addi_metrics(M_fold_5_T)
print(M_fold_5_T)
'''

##### Cellpose Scratch #####
cellpose_scratch_path = f"{result_path}/Cellpose_Scratch"
# Cellpose Dataset
'''
scratch_cellpose_test = np.load(f"{cellpose_scratch_path}/Cellpose/metrics_Cellpose.npy")
scratch_cellpose_test = calculate_addi_metrics(scratch_cellpose_test)
print(scratch_cellpose_test)

scratch_testis_test = np.load(f"{cellpose_scratch_path}/Cellpose/metrics_Testis.npy")
scratch_testis_test = calculate_addi_metrics(scratch_testis_test)
print(scratch_testis_test)
'''
# Testis Dataset
''' 
s_fold_1_path_T = np.load(f"{cellpose_scratch_path}/Testis/None_metrics_Fold_1.npy")
s_fold_1_path_T = calculate_addi_metrics(s_fold_1_path_T)
print(s_fold_1_path_T)

s_fold_2_path_T = np.load(f"{cellpose_scratch_path}/Testis/None_metrics_Fold_2.npy")
s_fold_2_path_T = calculate_addi_metrics(s_fold_2_path_T)
print(s_fold_2_path_T)

s_fold_3_path_T = np.load(f"{cellpose_scratch_path}/Testis/None_metrics_Fold_3.npy")
s_fold_3_path_T = calculate_addi_metrics(s_fold_3_path_T)
print(s_fold_3_path_T)

s_fold_4_path_T = np.load(f"{cellpose_scratch_path}/Testis/None_metrics_Fold_4.npy")
s_fold_4_path_T = calculate_addi_metrics(s_fold_4_path_T)
print(s_fold_4_path_T)
s_fold_5_path_T = np.load(f"{cellpose_scratch_path}/Testis/None_metrics_Fold_5.npy")
s_fold_5_path_T = calculate_addi_metrics(s_fold_5_path_T)
print(s_fold_5_path_T)
'''

# Mix Dataset
'''
s_fold_1_path_Mix = np.load(f"{cellpose_scratch_path}/Mix/None_metrics_Fold_1.npy")
s_fold_1_path_Mix = calculate_addi_metrics(s_fold_1_path_Mix)
print(s_fold_1_path_Mix)

s_fold_2_path_Mix = np.load(f"{cellpose_scratch_path}/Mix/None_metrics_Fold_2.npy")
s_fold_2_path_Mix = calculate_addi_metrics(s_fold_2_path_Mix)
print(s_fold_2_path_Mix)

#s_fold_3_path_Mix = np.load(f"{cellpose_scratch_path}/Mix/None_metrics_Fold_3.npy")
#s_fold_3_path_Mix = calculate_addi_metrics(s_fold_3_path_Mix)
#print(s_fold_3_path_Mix)

s_fold_4_path_Mix = np.load(f"{cellpose_scratch_path}/Mix/None_metrics_Fold_4.npy")
s_fold_4_path_Mix = calculate_addi_metrics(s_fold_4_path_Mix)
print(s_fold_4_path_Mix)

s_fold_5_path_Mix = np.load(f"{cellpose_scratch_path}/Mix/None_metrics_Fold_5.npy")
s_fold_5_path_Mix = calculate_addi_metrics(s_fold_5_path_Mix)
print(s_fold_5_path_Mix)
'''
##### DDT #####
'''
# Cellpose Dataset
path = "C:/Users/Tobias/Desktop/U-Net_Test/train/img/metrics_Fold_Test.npy"
path_test = "C:/Users/Tobias/Desktop/U-Net_Test/test/img"

for dateiname in os.listdir(path_test):
    if dateiname.endswith(".png"):
        bildpfad = os.path.join(path_test, dateiname)
        with Image.open(bildpfad) as bild:
            print(f"{dateiname}: {bild.size}")
array = np.load(path)
print(array)

# Testis Dataset 

# Mix Dataset
'''