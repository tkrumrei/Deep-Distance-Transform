# imports
import subprocess
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from my_cellpose import core, io, models, metrics


######################
# calculate_metrics(): calculates aggregated IoU and precision Score from cellpose for the masks
def calculate_metrics(true_masks, pred_masks):
    iou = metrics.aggregated_jaccard_index(true_masks, pred_masks) # output array
    average_iou = np.mean(iou)
    precision, n_true_p, n_false_p, n_false_n = metrics.average_precision(true_masks, pred_masks)
    average_precision = np.mean(precision)
    average_n_true_p = np.mean(n_true_p)
    average_n_false_p = np.mean(n_false_p)
    average_n_false_n = np.mean(n_false_n)

    print(f"Durchschnittliche IoU: {average_iou}")
    print(average_precision)
    return average_iou, average_precision, average_n_true_p, average_n_false_p, average_n_false_n


######################
# cellpose_eval(): Evaluates the Best Cellpose model on both datasets with IoU and Dice Score
def cellpose_eval(model_path, test_path, test_path_2, diam_labels, model_name, model_type):
    print(f"Evaluation von {model_name} startet")
    full_model_path = f"{model_path}models/{model_name}"
    use_GPU = core.use_gpu()
    model = models.CellposeModel(gpu=use_GPU, pretrained_model = full_model_path)


    ##### Cellpose test set #####
    output = io.load_train_test_data(test_path, image_filter = "_img")
    test_data, test_labels = output[:2]

    # run model on test images
    masks = model.eval(test_data, 
                    channels=[0, None],
                    diameter=diam_labels)[0]


    average_iou, precision, n_true_p, n_false_p, n_false_n = calculate_metrics(test_labels, masks)


    metrics_path = f"{model_path}models/{model_type}_metrics_Cellpose.npy"

    metrics = np.array([average_iou, precision, n_true_p, n_false_p, n_false_n])
    
    np.save(metrics_path, metrics)
    print("Cellpose Test ist fertig")
    
    ##### Testis Test set#####
    output = io.load_train_test_data(test_path_2, image_filter = "_img")
    test_data_2, test_labels_2 = output[:2]

    masks_2 = model.eval(test_data_2,
                        channels=[0, None],
                        diameter=diam_labels)[0]

    average_iou, precision, n_true_p, n_false_p, n_false_n = calculate_metrics(test_labels_2, masks_2)


    metrics_2_path = f"{model_path}models/{model_type}metrics_Testis.npy"

    metrics_2 = np.array([average_iou, precision, n_true_p, n_false_p, n_false_n])
    np.save(metrics_2_path, metrics_2)
    print("Testis Test ist fertig")

######################
# cellpose_train(): retrains cellpose model "model_name" 
def cellpose_train(train_path, test_path, test_path_2, model_type, model_name):
    print(f"Cellpose Model {model_type} Train startet für {model_name}.")
    logger = io.logger_setup()

    use_GPU = core.use_gpu()
    model = models.CellposeModel(gpu=use_GPU, model_type = model_type)

    channels = [0, None]

    output = io.load_train_test_data(train_path, image_filter="_img")
    train_data, train_labels, _, _, _, _ = output

    new_model_path = model.train(train_data, train_labels,
                                 channels=channels, 
                                 save_path=train_path,
                                 save_each = True,
                                 save_every=1,
                                 n_epochs=100,
                                 learning_rate=0.1, 
                                 weight_decay=0.0001,
                                 model_name=model_name)
    
    diam_labels = model.diam_labels.copy()
    
    cellpose_eval(train_path, test_path, test_path_2, diam_labels, model_name, model_type)

######################
# Function calls
palma_path = "/scratch/tmp/tkrumrei/Cellpose_Model"

##### Train on Datasets
# Cellpose Dataset
# cellpose_train("D:/Datasets/Cellpose_Model/Cellpose/Train/", "D:/Datasets/Cellpose_Model/Cellpose/Test_Cellpose/", "D:/Datasets/Cellpose_Model/Cellpose/Test_Testis/", None, "Cellpose_Model")
# Testis Dataset
cellpose_train(f"{palma_path}/Testis/Fold_1/", f"{palma_path}/Testis/Test_Cellpose/", f"{palma_path}/Testis/Test_Testis/", "cyto", "Cyto_Testis_Fold_1_Model")
cellpose_train(f"{palma_path}/Testis/Fold_2/", f"{palma_path}/Testis/Test_Cellpose/", f"{palma_path}/Testis/Test_Testis/", "cyto", "Cyto_Testis_Fold_2_Model")
cellpose_train(f"{palma_path}/Testis/Fold_3/", f"{palma_path}/Testis/Test_Cellpose/", f"{palma_path}/Testis/Test_Testis/", "cyto", "Cyto_Testis_Fold_3_Model")
cellpose_train(f"{palma_path}/Testis/Fold_4/", f"{palma_path}/Testis/Test_Cellpose/", f"{palma_path}/Testis/Test_Testis/", "cyto", "Cyto_Testis_Fold_4_Model")
cellpose_train(f"{palma_path}/Testis/Fold_5/", f"{palma_path}/Testis/Test_Cellpose/", f"{palma_path}/Testis/Test_Testis/", "cyto", "Cyto_Testis_Fold_5_Model")
# Mix Dataset
#cellpose_train("D:/Datasets/Cellpose_Model/Mix/Fold_1/", "D:/Datasets/Cellpose_Model/Mix/Test_Cellpose/", "D:/Datasets/Cellpose_Model/Mix/Test_Testis/", None, "Cellpose_Mix_Fold_1_Model")
#cellpose_train("D:/Datasets/Cellpose_Model/Mix/Fold_2/", "D:/Datasets/Cellpose_Model/Mix/Test_Cellpose/", "D:/Datasets/Cellpose_Model/Mix/Test_Testis/", None, "Cellpose_Mix_Fold_2_Model")
#cellpose_train("D:/Datasets/Cellpose_Model/Mix/Fold_3/", "D:/Datasets/Cellpose_Model/Mix/Test_Cellpose/", "D:/Datasets/Cellpose_Model/Mix/Test_Testis/", None, "Cellpose_Mix_Fold_3_Model")
#cellpose_train("D:/Datasets/Cellpose_Model/Mix/Fold_4/", "D:/Datasets/Cellpose_Model/Mix/Test_Cellpose/", "D:/Datasets/Cellpose_Model/Mix/Test_Testis/", None, "Cellpose_Mix_Fold_4_Model")
#cellpose_train("D:/Datasets/Cellpose_Model/Mix/Fold_5/", "D:/Datasets/Cellpose_Model/Mix/Test_Cellpose/", "D:/Datasets/Cellpose_Model/Mix/Test_Testis/", None, "Cellpose_Mix_Fold_5_Model")
