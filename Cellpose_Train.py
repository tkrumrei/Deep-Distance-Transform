import subprocess
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from glob import glob
from sklearn.metrics import jaccard_score, f1_score


def calculate_metrics(true_masks, pred_masks):
    true_masks = (true_masks > 0).astype(np.uint8).flatten()
    pred_masks = (pred_masks > 0).astype(np.uint8).flatten()

    iou = jaccard_score(true_masks, pred_masks)
    dice = f1_score(true_masks, pred_masks)

    return iou, dice


def normalize_images(images): 
    # Konvertiert zu float, falls noch nicht geschehen
    normalized_images = []
    for image in images:
        norm_image = np.array(image, dtype=np.float32)
        norm_image /= 65535.0 # Normalisiere basierend auf dem möglichen Maximalwert für uint8
        normalized_images.append(norm_image)
    return normalized_images

def display_images_and_masks(images, masks, title=""):
    """Anzeigefunktion für Bilder und Masken."""
    n = len(images)  # Anzahl der Bilder/Masken
    plt.figure(figsize=(12, 3*n))
    for i in range(n):
        
        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(images[i])
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(masks[i], cmap='jet')  # Verwende einen Farbverlauf, um die Maske hervorzuheben
        plt.title("Generated Mask")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def cellpose_train(train_path, test_path, test_path_2, model_type, model_name):
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
                                 n_epochs=10,
                                 learning_rate=0.1, 
                                 weight_decay=0.0001,
                                 model_name=model_name)
    
    diam_labels = model.diam_labels.copy()
    
    output = io.load_train_test_data(test_path, image_filter = "_img")
    test_data, test_labels = output[:2]

    # run model on test images
    masks = model.eval(test_data, 
                    channels=[0, None],
                    diameter=diam_labels)[0]

    # Beispielaufruf der Funktion
    # Angenommen, test_labels und masks sind die Listen der wahren Masken und der vorhergesagten Masken
    iou_scores = []
    dice_scores = []

    for true_mask, pred_mask in zip(test_labels, masks):
        iou, dice = calculate_metrics(true_mask, pred_mask)
        iou_scores.append(iou)
        dice_scores.append(dice)

    average_iou = sum(iou_scores) / len(iou_scores)
    average_dice = sum(dice_scores) / len(dice_scores)

    print(f"Average IoU: {average_iou:.4f}")
    print(f"Average Dice Score: {average_dice:.4f}")

    display_images_and_masks(normalize_images(test_data), masks, title="Test Set 1 Results")
    
    output = io.load_train_test_data(test_path_2, image_filter = "_img")
    test_data_2, test_labels_2 = output[:2]

    masks2 = model.eval(test_data_2,
                        channels=[0, None],
                        diameter=diam_labels)[0]
    
    iou_scores_2 = []
    dice_scores_2 = []

    for true_mask, pred_mask in zip(test_labels_2, masks2):
        iou, dice = calculate_metrics(true_mask, pred_mask)
        iou_scores_2.append(iou)
        dice_scores_2.append(dice)

    average_iou = sum(iou_scores_2) / len(iou_scores_2)
    average_dice = sum(dice_scores_2) / len(dice_scores_2)

    print(f"Average IoU: {average_iou:.4f}")
    print(f"Average Dice Score: {average_dice:.4f}")

    display_images_and_masks(test_data_2, masks2, title="Test Set 2 Results")
    


cellpose_train("C:/Users/Tobias/Desktop/test/train/", "C:/Users/Tobias/Desktop/test/test1/", "C:/Users/Tobias/Desktop/test/test2/", None, "Cellpose_Model")



