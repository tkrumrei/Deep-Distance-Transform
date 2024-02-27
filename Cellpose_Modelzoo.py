# imports
import numpy as np
#import matplotlib.pyplot as plt
from my_cellpose import core, io, models, metrics
from PIL import Image
import os

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

    print(iou)
    print("")
    print(f"Durchschnittliche IoU: {average_iou}")
    print(f"Durchschnittliche Präzision: {average_precision}")
    return average_iou, average_precision, average_n_true_p, average_n_false_p, average_n_false_n

######################
# cellpose_modelzoo_eval(): Evaluates the Best Cellpose model on both datasets with IoU and Dice Score
def cellpose_modelzoo_eval(test_path, test_path_2, diam_labels, model_type):

    use_GPU = core.use_gpu()
    model = models.CellposeModel(gpu=use_GPU, model_type=model_type)
    print("Nuclei Model wird zur Evaluierung genommen")


    ##### Cellpose test set #####
    output = io.load_train_test_data(test_path, image_filter = "_img")
    test_data, test_labels = output[:2]

    # run model on test images
    masks = model.eval(test_data, 
                    channels=[0, None],
                    diameter=diam_labels)[0]
    
 

    true_masks = []

    for filename in os.listdir(test_path):
        if filename.endswith('_masks.png'):
            file_path = os.path.join(test_path, filename)
            mask_image = Image.open(file_path)
            true_masks.append(np.array(mask_image))

    # only for visualization
    '''
    # display results
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(true_masks[60], cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(masks[60], cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('predicted mask')

    fig.tight_layout()

    plt.show()
    '''
    average_iou, precision, n_true_p, n_false_p, n_false_n = calculate_metrics(true_masks, masks)

    ##### local #####
    #metrics_path = f"D:/Bachelorarbeit/Ergebnisse/{model_type}_metrics_Cellpose.npy"
    ##### Palma #####
    metrics_path = f"/scratch/tmp/tkrumrei/Ergebnisse/{model_type}_metrics_Cellpose.npy"

    metrics = np.array([average_iou, precision, n_true_p, n_false_p, n_false_n])
    
    np.save(metrics_path, metrics)
    print("Cellpose Test ist fertig")

    # only for visual control
    # display_images_and_masks(normalize_images(test_data), masks, title="Test Set 1 Results")
    
    ##### Testis Test set#####
    output = io.load_train_test_data(test_path_2, image_filter = "_img")
    test_data_2, test_labels_2 = output[:2]

    masks_2 = model.eval(test_data_2,
                        channels=[0, None],
                        diameter=diam_labels)[0]
    
    true_masks_2 = []

    for filename in os.listdir(test_path_2):
        if filename.endswith('_masks.png'):
            file_path = os.path.join(test_path_2, filename)
            mask_image = Image.open(file_path)
            true_masks.append(np.array(mask_image))

    average_iou, precision, n_true_p, n_false_p, n_false_n = calculate_metrics(true_masks_2, masks_2)

    metrics_2_path = f"D:/Bachelorarbeit/Ergebnisse/{model_type}_metrics_Testis.npy"
    metrics_2_path = f"/scratch/tmp/tkrumrei/Ergebnisse/{model_type}_metrics_Testis.npy"

    metrics_2 = np.array([average_iou, precision, n_true_p, n_false_p, n_false_n])
    np.save(metrics_2_path, metrics_2)
    print("Testis Test ist fertig")

    # only for visual control
    # display_images_and_masks(test_data_2, masks2, title="Test Set 2 Results")

##### Lokal #####
#cellpose_modelzoo_eval("D:/Datasets/Cellpose_Model/Cellpose/Test_Cellpose/", "D:/Datasets/Cellpose_Model/Cellpose/Test_Testis/", None, "nuclei")
#cellpose_modelzoo_eval("D:/Datasets/Cellpose_Model/Cellpose/Test_Cellpose/", "D:/Datasets/Cellpose_Model/Cellpose/Test_Testis/", None, "cyto")
##### Palma #####
cellpose_modelzoo_eval("/scratch/tmp/tkrumrei/Cellpose_Model/Cellpose/Test_Cellpose/", "/scratch/tmp/tkrumrei/Cellpose_Model/Cellpose/Test_Testis/", None, "nuclei")
cellpose_modelzoo_eval("/scratch/tmp/tkrumrei/Cellpose_Model/Cellpose/Test_Cellpose/", "/scratch/tmp/tkrumrei/Cellpose_Model/Cellpose/Test_Testis/", None, "cyto")