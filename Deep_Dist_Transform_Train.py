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

####################
# DDT_Train(): Train Deep Distance Transform model and save best model
def DDT_Train(train_path):
    train_image_folder = f"{train_path}/img"
    train_dt_folder = f"{train_path}/distance_transform"
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=2e-4)

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
    print("Fertig!")

DDT_Train("D:/Datasets/Testis_Model/Cellpose/Train")
