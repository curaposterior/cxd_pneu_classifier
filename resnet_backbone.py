import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, ResNet
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_score, recall_score, f1_score

import time
from datetime import datetime
from functools import wraps


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Took {(end_time - start_time)/60:.2f} minutes to execute")
        return result

    return wrapper


TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((336, 336)),
        transforms.RandomResizedCrop(336, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# test and validation
TEST_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

BATCH_SIZE = 64


def get_resnet_model(num_of_classes: int = 2) -> ResNet:
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_of_classes)
    return resnet


@timing_decorator
def train_resnet(
    net: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    loss_fn,
    scheduler=None,
    epochs: int = 10,
    device: str = "cpu",
):
    net = net.to(device)
    writer = SummaryWriter()

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # check this
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 20 == 19:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Step [{i + 1}/{len(train_loader)}], "
                    f"Loss: {running_loss / 20:.3f}, "
                    f"Accuracy: {100 * correct / total:.2f}%"
                )
                running_loss = 0.0
        
        val_loss, val_acc = test_resnet(net, val_loader, loss_fn, device)

        if scheduler:
            scheduler.step(val_loss)

        writer.add_scalar("Loss/train", val_loss, epoch)
        writer.add_scalar("Accuracy/train", val_acc, epoch)
        print(f"Validation: loss = {val_loss:.3f}, accuracy = {val_acc:.2f}%")
        writer.close()

    return net


def test_resnet(
    net: nn.Module, test_loader: DataLoader, loss_fn=None, device: str = "cuda"
):
    net = net.to(device)
    net.eval()

    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            if loss_fn:
                test_loss += loss_fn(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader) if loss_fn else 0

    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    return avg_loss, accuracy


if __name__ == "__main__":
    root_dir = "dataset/chest_xray/chest_xray/"
    train_data_path = "train"
    test_data_path = "test"
    val_data_path = "val"

    train_dataset = torchvision.datasets.ImageFolder(
        root=root_dir + train_data_path, transform=TRAIN_TRANSFORMS
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=root_dir + test_data_path, transform=TEST_TRANSFORMS
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=root_dir + val_data_path, transform=TEST_TRANSFORMS
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"training dataset size: {len(train_loader.dataset)}")
    print(f"testing dataset size: {len(test_loader.dataset)}")
    print(f"validation dataset size: {len(val_loader.dataset)}")

    classes = train_dataset.classes
    print(train_dataset.class_to_idx)
    network = get_resnet_model(2)
    network = network.to(device)

    class_counts = [sum(1 for _, label in train_dataset if label == c) 
                   for c in range(len(classes))]
    total_samples = sum(class_counts)
    class_weights = torch.FloatTensor([total_samples / (len(classes) * c) 
                                     for c in class_counts]).to(device)

    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(network.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
    )

    epochs = 25

    trained_model = train_resnet(
        net=network,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_function,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
    )

    print("\nFINAL testing on test dataset")
    test_resnet(trained_model, test_loader, loss_function, device)
    torch.save(network, f'model_e{epochs}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.pt')