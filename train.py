import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from omegaconf import OmegaConf

from model.hybrid_model import HybridModel

# ------------------------------
# Training Loop
# ------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


# ------------------------------
# Main Function
# ------------------------------
def main(config_path="configs/default.yaml"):
    # Load config
    config = OmegaConf.load(config_path)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Replace with satellite dataset later
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4)

    # Model
    model = HybridModel(config.model).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr)

    # MLflow setup
    mlflow.set_tracking_uri(config.logging.mlflow_tracking_uri)
    mlflow.set_experiment(config.logging.experiment_name)

    with mlflow.start_run():
        mlflow.log_params({
            "cnn": config.model.cnn,
            "transformer": config.model.transformer,
            "fusion": config.model.fusion,
            "batch_size": config.training.batch_size,
            "lr": config.training.lr,
            "epochs": config.training.epochs
        })

        # Training loop
        for epoch in range(config.training.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1}/{config.training.epochs} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

        # Save final model
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
