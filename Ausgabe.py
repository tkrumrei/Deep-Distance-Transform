import numpy as np

result_path = "D:/Bachelorarbeit/Ergebnisse"

##### Cellpose Modelzoo #####
modelzoopath = f"{result_path}/Cellpose_Modelzoo"

# Nuclei Model
nuclei_cellpose_path = f"{modelzoopath}/nuclei_metrics_Cellpose.npy"
nuclei_testis_path = f"{modelzoopath}/nuclei_metrics_Testis.npy"

nuclei_cellpose = np.load(nuclei_cellpose_path)
nuclei_testis = np.load(nuclei_testis_path)

# print(nuclei_cellpose)
# print(nuclei_testis)

# Cyto Model
cyto_cellpose_path = f"{modelzoopath}/cyto_metrics_Cellpose.npy"
cyto_testis_path = f"{modelzoopath}/cyto_metrics_Testis.npy"

cyto_cellpose = np.load(cyto_cellpose_path)
cyto_testis = np.load(cyto_testis_path)

# print(cyto_cellpose)
# print(cyto_testis)

arrays = [nuclei_cellpose, nuclei_testis, cyto_cellpose, cyto_testis]
rounded = [np.round(arr, 5) for arr in arrays]
print(rounded)