import subprocess
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from glob import glob

def run_cellpose_training(train_path, test_path, model):

    base_command = "python -m cellpose"

    # train parameters
    training_command = f"--train --dir {train_path} --test_dir {test_path} --pretrained_model {model} --chan 2 --chan2 0 --diam_mean 17.0 --img_filter _img --mask_filter _masks --n_epochs 10 --verbose"

    full_command = f"{base_command} {training_command}"

    subprocess.run(full_command, shell=True, check=True)

# run_cellpose_training("D:/Datasets/Cellpose_Model/Cellpose/Train/", "D:/Datasets/Cellpose_Model/Cellpose/Test_Cellpose/", "None")


def cellpose_train(train_path, test_path, model_type, chan, chan2, model_name):
    logger = io.logger_setup()

    use_GPU = core.use_gpu()
    model = models.CellposeModel(gpu=use_GPU, pretrained_model = model_type)

    channels = [chan, chan2]

    output = io.load_train_test_data(train_path, test_path, image_filter = "_img")
    train_data, train_labels, _, test_data, test_labels, _ = output

    new_model_path = model.train(train_data, train_labels,
                                 test_data = test_data,
                                 test_labels = test_labels,
                                 channels=channels, 
                                 save_path=train_path,
                                 save_every=1,
                                 n_epochs=100,
                                 learning_rate=0.1, 
                                 weight_decay=0.0001,
                                 model_name=model_name)
    
    diam_labels = model.diam_labels.copy()

    output = io.load_train_test_data(test_path, image_filter = "_img", mask_filter ="_masks")
    test_data, test_labels = output[:2]

    # run model on test images
    masks = model.eval(test_data, 
                    channels=[chan, chan2],
                    diameter=diam_labels)[0]

    # check performance using ground truth labels
    ap = metrics.average_precision(test_labels, masks)[0]
    print('')
    print(f'>>> average precision at iou threshold 0.5 = {ap[:,0].mean():.3f}')

cellpose_train("D:/Datasets/Cellpose_Model/Cellpose/Train/", "D:/Datasets/Cellpose_Model/Cellpose/Test_Cellpose/", "None", 2, 0, "Cellpose_Model")



