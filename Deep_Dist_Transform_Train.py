'''
Deep Distance Transform Paper:
- Batch Größe 25
- Adam Optimizer mit Parametern (betas, epsilon, weight decay)
- Trainingsdauer 10 Epochen
'''
# Imports
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from UNet import UNet
from PIL import Image
from my_cellpose import core, io, models, metrics
import Segmentation
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    def __init__(self, image_folder, dt_folder, weights_folder, transform=None):
        self.image_folder = image_folder
        self.dt_folder = dt_folder
        self.weights_folder = weights_folder
        self.transform = transform
        self.images = [f for f in os.listdir(image_folder) if f.endswith('.png')] 
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.images[index])
        dt_path = os.path.join(self.dt_folder, self.images[index].replace("_img.png", "_dt.npy"))
        weight_path = os.path.join(self.weights_folder, self.images[index].replace("_img.png", "_w.npy"))

        image = Image.open(image_path).convert("RGB")
        dist_transform = np.load(dt_path)
        weight = np.load(weight_path)

        if self.transform is not None:
                image = self.transform(image)  # transform of image

        dist_transform = Image.fromarray(np.uint8(dist_transform*255))  # skaling
        weight = Image.fromarray(np.uint8(weight*255))  # skaling
        
        dist_transform = self.transform(dist_transform) 
        weight = self.transform(weight)

        return image, dist_transform, weight

####################
# WeightedEuclideanLoss(): returns the mean weighted Euclidean Loss for the Epoche
class WeightedEuclideanLoss(nn.Module):
    def __init__(self):
        super(WeightedEuclideanLoss, self).__init__()
    
    def forward(self, predictions, targets, weights):
        loss = (weights * (predictions - targets) ** 2).mean()
        return loss

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(), # changes T to Tensor
])

test_transform = T.Compose([
    T.ToTensor(),  # changes T to Tensor
])

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
    print(f"Durchschnittliche Präzision {average_precision}")
    return average_iou, average_precision, average_n_true_p, average_n_false_p, average_n_false_n

######################
# load_true_masks():
def load_true_masks(masks_folder):
    mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.png')]
    masks = []
    for file_name in mask_files:
        file_path = os.path.join(masks_folder, file_name)
        mask = Image.open(file_path)
        mask_array = np.array(mask)
        masks.append(mask_array)
    return masks

######################
# DDT_predict():
def DDT_predict(train_image_folder, test_path, test_path_2, fold_n):

    test_mask_folder = f"{test_path}/masks"
    test_2_mask_folder = f"{test_path_2}/masks"
    # Modell laden
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(f"{train_image_folder}/Best_Model.pth"))
    model.eval()

    if len(test_path_2) > 1:
        ##### Test Cellpose #####
        print("Cellpose Test hat begonnen")
        test_dataset = ImageDataset(
            image_folder=f"{test_path}/img",
            dt_folder=f"{test_path}/dt",
            weights_folder=f"{test_path}/weights",
            transform=test_transform
        )

        # DataLoader for Cellpose Test
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        masks_list = []
        true_masks_list = load_true_masks(test_mask_folder)
        with torch.no_grad():
            for i, (image, _, _) in enumerate(test_loader):
                output = model(image)
                predicted_dt_np = output[0].squeeze().cpu().detach().numpy()
                mask = Segmentation.make_mask(predicted_dt_np, 1051.0) # average area of Cellpose
                masks_list.append(mask)

        average_iou, precision, n_true_p, n_false_p, n_false_n = calculate_metrics(true_masks_list, masks_list)


        metrics = np.array([average_iou, precision, n_true_p, n_false_p, n_false_n])
        
        np.save(f"{train_image_folder}/metrics_Cellpose.npy", metrics)
        print("metrics_Cellpose.npy ist fertig")
        print("-----------------------------------")

        ##### Test Testis #####
        print("Testis Test hat begonnen")
        test_dataset_2 = ImageDataset(
            image_folder=f"{test_path_2}/img",
            dt_folder=f"{test_path_2}/distance_transform",
            weights_folder=f"{test_path_2}/weights",
            transform=test_transform
        )

        # DataLoader for Testis Test
        test_loader = DataLoader(test_dataset_2, batch_size=1, shuffle=False)

        masks_2_list = []
        true_masks_2_list = load_true_masks(test_2_mask_folder)
        with torch.no_grad():
            for i, (image, _, _) in enumerate(test_loader):
                output = model(image)
                predicted_dt_np = output[0].squeeze().cpu().detach().numpy()
                mask = Segmentation.make_mask(predicted_dt_np, 300.0) # average area of Testis
                masks_2_list.append(mask)
                print(f"Erstellung für Bild {i}")

        average_iou, precision, n_true_p, n_false_p, n_false_n = calculate_metrics(true_masks_2_list, masks_2_list)


        metrics_2 = np.array([average_iou, precision, n_true_p, n_false_p, n_false_n])
        
        np.save(f"{train_image_folder}/metrics_Testis.npy", metrics_2)
        print("metrics_Testis.npy ist fertig")
    else:
        ##### Test Fold #####
        print(f"{fold_n} Test hat begonnen")
        test_dataset = ImageDataset(
            image_folder=f"{test_path}/img",
            dt_folder=f"{test_path}/dt",
            weights_folder=f"{test_path}/weights",
            transform=test_transform
        )

        # DataLoader for Cellpose Test
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        masks_list = []
        true_masks_list = load_true_masks(test_mask_folder)
        with torch.no_grad():
            for i, (image, _, _) in enumerate(test_loader):
                output = model(image)
                predicted_dt_np = output[0].squeeze().cpu().detach().numpy()
                mask = Segmentation.make_mask(predicted_dt_np, 1051.0) # average area of Cellpose
                masks_list.append(mask)

        average_iou, precision, n_true_p, n_false_p, n_false_n = calculate_metrics(true_masks_list, masks_list)


        metrics = np.array([average_iou, precision, n_true_p, n_false_p, n_false_n])
        print(metrics)
        
        np.save(f"{train_image_folder}/metrics_{fold_n}.npy", metrics)
        print(f"metrics_{fold_n}.npy ist fertig")
    print("-------------Ende---------------")

####################
# DDT_Train(): Train Deep Distance Transform model and save best model
def DDT_train(train_path, test_path, test_path_2, fold_n):
    train_image_folder = f"{train_path}/img"
    train_dt_folder = f"{train_path}/dt"
    train_weights_folder = f"{train_path}/weights"

    # save dataset in dataset
    dataset = ImageDataset(
        image_folder=train_image_folder,
        dt_folder=train_dt_folder,
        weights_folder= train_weights_folder,
        transform=transform
    )

    # Data Loader with mini-batch size 25
    train_loader = DataLoader(dataset, batch_size=25, shuffle=True)

    # model, loss function and Adam optimizer with parameters from paper
    model = UNet(in_channels=3, out_channels=1)
    loss_fn = WeightedEuclideanLoss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.1, 0.999), eps=1e-8, weight_decay=2e-4)

    # training
    num_epochs = 10
    smallest_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_batches = len(train_loader)

        for inputs, targets, weights in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value = loss_fn(outputs, targets, weights)
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()
        
        average_loss = total_loss / total_batches
        print(f"Epoch{epoch} fertig, Durchschnittlicher Loss: {average_loss}")

        if average_loss < smallest_loss:
            smallest_loss = average_loss 
            model_save_path = os.path.join(train_image_folder, f"Best_Model.pth") 
            torch.save(model.state_dict(), model_save_path)
            print(f"Modell von Epoche {epoch} gespeichert")
    print("Training ist Fertig!")
    print("-----------------------------------")
    print("Tests beginnen")
    DDT_predict(train_image_folder, test_path, test_path_2, fold_n)

# local function call with small dataset for testing
#DDT_train("C:/Users/Tobias/Desktop/U-Net_Test/train", "C:/Users/Tobias/Desktop/U-Net_Test/test", "", "Test")
#DDT_predict("D:/Bachelorarbeit/Ergebnisse/DDT_Model", "C:/Users/Tobias/Desktop/U-Net_Test/test", "", "Test")
##### Function Calls #####
# Palma Path
palma_path = "/scratch/tmp/tkrumrei/DDT_Model"
'''
# Cellpose
DDT_train(f"{palma_path}/Cellpose/Train", f"{palma_path}/Test_Cellpose", f"{palma_path}/Test_Testis", "")
#DDT_predict(f"{palma_path}/Cellpose/Train/img", f"{palma_path}/Test_Testis", "", "")
# Testis
DDT_train(f"{palma_path}/Testis/Fold_1/Train", f"{palma_path}/Testis/Fold_1/Validate", "", "Fold 1")
DDT_train(f"{palma_path}/Testis/Fold_2/Train", f"{palma_path}/Testis/Fold_2/Validate", "", "Fold 2")
DDT_train(f"{palma_path}/Testis/Fold_3/Train", f"{palma_path}/Testis/Fold_3/Validate", "", "Fold 3")
DDT_train(f"{palma_path}/Testis/Fold_4/Train", f"{palma_path}/Testis/Fold_4/Validate", "", "Fold 4")
DDT_train(f"{palma_path}/Testis/Fold_5/Train", f"{palma_path}/Testis/Fold_5/Validate", "", "Fold 5")
# Mix
DDT_train(f"{palma_path}/Mix/Fold_1/Train", f"{palma_path}/Mix/Fold_1/Validate", "", "Fold 1")
DDT_train(f"{palma_path}/Mix/Fold_2/Train", f"{palma_path}/Mix/Fold_2/Validate", "", "Fold 2")
DDT_train(f"{palma_path}/Mix/Fold_3/Train", f"{palma_path}/Mix/Fold_3/Validate", "", "Fold 3")
DDT_train(f"{palma_path}/Mix/Fold_4/Train", f"{palma_path}/Mix/Fold_4/Validate", "", "Fold 4")
DDT_train(f"{palma_path}/Mix/Fold_5/Train", f"{palma_path}/Mix/Fold_5/Validate", "", "Fold 5")
'''