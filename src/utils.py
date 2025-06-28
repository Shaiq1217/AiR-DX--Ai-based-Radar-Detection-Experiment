import torch
import numpy as np
import pandas as pd
def get_conv_output(model, input_shape):
    """Compute the flattened output size of a CNN model after conv layers."""
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)  # e.g. (1, 224, 224)
        x = model._forward_conv(x)
        return int(np.prod(x.size()))

import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path, save_path):
    # Load CSV
    df = pd.read_csv(csv_path)
    df['epoch'] = df['epoch'].astype(int)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    axs[0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    axs[0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='x')
    axs[0].set_title('Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy
    axs[1].plot(df['epoch'], df['train_acc'], label='Train Accuracy', marker='o')
    axs[1].plot(df['epoch'], df['val_acc'], label='Val Accuracy', marker='x')
    axs[1].set_title('Accuracy over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("out/training_metrics_plot.png")  # Save the figure
    plt.show()



    