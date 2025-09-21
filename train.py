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

from data.dataset import XView2Dataset
from model.siamese_model import SiameseNetwork 
from model.losses import ContrastiveLoss 

from torch.optim.lr_scheduler import StepLR
from utils.evaluation import calculate_metrics



# # ------------------------------
# # Training Loop
# # ------------------------------
# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss, correct, total = 0, 0, 0

#     for images, labels in dataloader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         _, preds = torch.max(outputs, 1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#     return total_loss / len(dataloader), correct / total


# def validate(model, dataloader, criterion, device):
#     model.eval()
#     total_loss, correct, total = 0, 0, 0

#     with torch.no_grad():
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             total_loss += loss.item()
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     return total_loss / len(dataloader), correct / total

# ------------------------------
# Updated Training Loop for Siamese Network
# ------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for pre_image, post_image, labels in dataloader:
        # Move data to the device
        pre_image, post_image, labels = pre_image.to(device), post_image.to(device), labels.to(device).float()

        optimizer.zero_grad()

        # Get the two outputs from the siamese network
        output1, output2 = model(pre_image, post_image)

        # Calculate loss
        loss = criterion(output1, output2, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for pre_image, post_image, labels in dataloader:
            pre_image, post_image, labels = pre_image.to(device), post_image.to(device), labels.to(device).float()

            output1, output2 = model(pre_image, post_image)
            loss = criterion(output1, output2, labels)
            running_loss += loss.item()
            
            # Note: Accuracy is harder to measure directly here.
            # You might calculate distance and compare to a threshold,
            # but for now, we'll focus on the validation loss.

    return running_loss / len(dataloader)


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

        # Using placeholder IDs for now
    train_ids = ["img1", "img2", "img3"] # Replace with your actual image IDs
    val_ids = ["img4", "img5"] # Replace with your actual image IDs


    train_dataset = XView2Dataset(
        images_dir=config.dataset.images_dir,
        labels_dir=config.dataset.labels_dir,
        image_ids=train_ids,
        transform=transform
    )
    val_dataset = XView2Dataset(
        images_dir=config.dataset.images_dir,
        labels_dir=config.dataset.labels_dir,
        image_ids=val_ids,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4)

    #   Model - Now using the SiameseNetwork 
    model = SiameseNetwork(config.model).to(device)

    # Model
    # model = HybridModel(config.model).to(device)

    # # Loss & Optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=config.training.lr)

    criterion = ContrastiveLoss(margin=config.training.loss_margin)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.training.lr, 
        weight_decay=config.training.get("weight_decay", 0.0)
    )

    scheduler = StepLR(
        optimizer, 
        step_size=config.training.scheduler.step_size, 
        gamma=config.training.scheduler.gamma
    )



    # MLflow setup need to modify the training loop to handle image pairs.
    mlflow.set_tracking_uri(config.logging.mlflow_tracking_uri)
    mlflow.set_experiment(config.logging.experiment_name)

    # with mlflow.start_run():
    #     mlflow.log_params({
    #         "cnn": config.model.cnn,
    #         "transformer": config.model.transformer,
    #         "fusion": config.model.fusion,
    #         "batch_size": config.training.batch_size,
    #         "lr": config.training.lr,
    #         "epochs": config.training.epochs
    #     })

    #     # Training loop
    #     for epoch in range(config.training.epochs):
    #         train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    #         val_loss, val_acc = validate(model, val_loader, criterion, device)

    #         print(f"Epoch {epoch+1}/{config.training.epochs} | "
    #               f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    #         # Log metrics
    #         mlflow.log_metrics({
    #             "train_loss": train_loss,
    #             "train_acc": train_acc,
    #             "val_loss": val_loss,
    #             "val_acc": val_acc
    #         }, step=epoch)

    #     # Save final model
    #     mlflow.pytorch.log_model(model, "model")

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))

        # Training loop
        for epoch in range(config.training.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            eval_metrics = calculate_metrics(
                model, val_loader, device, threshold=config.training.eval_threshold
            )

            scheduler.step()

            
            # val_loss = validate(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1}/{config.training.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val F1: {eval_metrics['f1_score']:.4f}")


            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metrics(eval_metrics, step=epoch)

        # Save final model
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()
