from torchvision import models, transforms
import torch.nn as nn

def ResnetModel(num_classes=3):
    model = models.resnet18(pretrained=True)
    
    # Convert grayscale to RGB if needed in dataloader
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model
