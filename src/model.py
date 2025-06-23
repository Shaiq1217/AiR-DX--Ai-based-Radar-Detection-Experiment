import torch
import torch.nn as nn
from src.utils import get_conv_output  # Assumes this computes output shape dynamically

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3, input_shape=(1, 224, 224)):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Compute conv output size
        conv_output_size = get_conv_output(self, input_shape)

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _forward_conv(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # -> (32, 112, 112)
        x = self.pool(self.relu(self.conv2(x)))  # -> (64, 56, 56)
        x = self.pool(self.relu(self.conv3(x)))  # -> (128, 28, 28)
        x = self.pool(self.relu(self.conv4(x)))  # -> (128, 14, 14)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
