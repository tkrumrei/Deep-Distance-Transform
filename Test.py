import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from Deep_Dist_Transform_Train import ImageDataset
from UNet import UNet  

def evaluation(test_folder):
    test_image_folder = f"{test_folder}/img"
    test_dt_folder = f"{test_folder}/distance_transform"
    test_weights_folder = f"{test_folder}/weights"

    # Schritt 1: Modellarchitektur definieren
    model = UNet(in_channels=3, out_channels=1)

    # Schritt 2: Modellzustand laden
    model_path = "D:/Datasets/Testis_Model/Cellpose/Train/img/unet_model_epoch_9.pth"
    model.load_state_dict(torch.load(model_path))

    # Schritt 3: Datensatz vorbereiten
    transform = transforms.Compose([
        # Definieren Sie hier Ihre Transformationen
        transforms.ToTensor(),
        # Fügen Sie weitere Transformationen hinzu, falls benötigt
    ])
    dataset = ImageDataset(
        image_folder=test_image_folder,
        dt_folder=test_dt_folder,
        weights_folder=test_weights_folder,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False)

    # Schritt 4: Modell evaluieren
    model.eval()

    # Schritt 5: Vorhersagen durchführen
    with torch.no_grad():  # Deaktiviert die Gradientenberechnung
        for inputs in dataloader:
            outputs = model(inputs)  # Erhalten Sie die Vorhersagen des Modells
            # Verarbeiten Sie hier die Ausgaben (z.B. Extrahieren von Vorhersagen, Berechnen von Metriken)
            print(outputs)



evaluation("D:/Datasets/Testis_Model/Cellpose/Test_Cellpose")