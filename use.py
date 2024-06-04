from test import Network
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from PIL import Image


def predict_image(image_path, model, device):
    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    output = model(image)

    _, predicted = torch.max(output, 1)

    return 'Pneumonia' if predicted.item() == 1 else 'Normal'


if __name__ == "__main__":
    image_path = 'pneumonia_photo.jpg'  # replace with your image file path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'model_e12_18_05_2024_19_18_20.pt'
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    print(predict_image(image_path, model, device))