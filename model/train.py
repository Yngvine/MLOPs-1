"""
Script to train a MobileNetV2 model on the Oxford-IIIT Pet dataset using transfer learning.
Logs experiments to MLflow.
"""
import os
import json
import argparse
from typing import cast
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import mlflow
from mlflow import pytorch as mlflow_pytorch

def train_model():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on Oxford-IIIT Pet dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # MLflow setup
    mlflow.set_experiment("OxfordPets_MobileNetV2")
    
    run_name = f"mobilenet_v2_lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "model": "mobilenet_v2",
            "dataset": "Oxford-IIIT Pet",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss"
        })

        # Data transformations
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Load dataset
        data_dir = './data/unpacked/'
        os.makedirs(data_dir, exist_ok=True)
                
        full_dataset = datasets.OxfordIIITPet(root=data_dir, split='trainval', download=False, transform=data_transforms['train'])
        
        # Get class names
        class_names = full_dataset.classes
        num_classes = len(class_names)
        
        # Log class labels
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(class_names, tmp)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="class_labels")
        os.remove(tmp_path)

        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
        
        # Update transform for validation set (though here they are the same, good practice to separate)
        # Note: random_split returns Subset, which doesn't have transform attribute directly accessible easily to change.
        # But since we used the same transform for both initially, it's fine. 
        # Ideally we would wrap the subset to apply different transforms, but for this lab, resizing to 224 is key.

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2),
            'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        }
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

        # Model setup
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        
        # Freeze feature extractor
        for param in model.features.parameters():
            param.requires_grad = False
            
        # Modify classifier
        # MobileNetV2 classifier is a Sequential with Dropout and Linear.
        # We replace the last Linear layer.
        # model.classifier[1] is the Linear layer.
        last_layer = cast(nn.Linear, model.classifier[1])
        num_ftrs = last_layer.in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

        # Training loop
        train_loss_history = []
        val_loss_history = []
        
        best_acc = 0.0

        print(f"Starting training on {device}...")
        
        for epoch in range(args.epochs):
            print(f'Epoch {epoch}/{args.epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = torch.tensor(0, device=device)

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Log metrics
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                mlflow.log_metric(f"{phase}_accuracy", epoch_acc.item(), step=epoch)
                
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                else:
                    val_loss_history.append(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc

        print(f'Best val Acc: {best_acc:.4f}')

        # Plot loss curves
        plt.figure()
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            tmp_path = tmp.name
        
        mlflow.log_artifact(tmp_path, artifact_path="plots")
        os.remove(tmp_path)

        # Register model
        # We use the same name "OxfordPetsMobileNetV2" for all registered models
        mlflow_pytorch.log_model(model, "model", registered_model_name="OxfordPetsMobileNetV2")

if __name__ == "__main__":
    train_model()
