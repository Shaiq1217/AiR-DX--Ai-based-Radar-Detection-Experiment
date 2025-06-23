from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

def get_test_loader():
    """Get the test DataLoader."""
    # Define the transformation to apply to each image
    transform = transforms.Compose([
        transforms.Grayscale(),        
        transforms.Resize((224, 224)),   
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.ImageFolder(root="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    classes = test_dataset.classes
    return test_loader, classes

def run_inferece(model, device='cpu'):
    """Run inference on the test dataset."""
    test_loader, classes = get_test_loader()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Evaluation
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nF1 Score:", f1_score(all_labels, all_preds, average='weighted'))
    print("Precision:", precision_score(all_labels, all_preds, average='weighted'))
    print("Recall:", recall_score(all_labels, all_preds, average='weighted'))