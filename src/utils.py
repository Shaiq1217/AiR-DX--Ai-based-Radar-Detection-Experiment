import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_conv_output(model, input_shape):
    """Compute the flattened output size of a CNN model after conv layers."""
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)  # e.g. (1, 224, 224)
        x = model._forward_conv(x)
        return int(np.prod(x.size()))


def plot_metrics(csv_path, save_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Plot metrics
    plt.figure(figsize=(8, 5))
    plt.plot(df['Noise Multiplier'], df['F1 Score'], label='F1 Score', marker='o')
    plt.plot(df['Noise Multiplier'], df['Precision'], label='Precision', marker='x')
    plt.plot(df['Noise Multiplier'], df['Recall'], label='Recall', marker='s')

    # Set y-axis range (e.g., from 0.6 to 1.05)
    plt.ylim(0.6, 1.05)

    # Add labels, title, and legend
    plt.xlabel("Noise Multiplier")
    plt.ylabel("Score")
    plt.title("Model Performance vs. Noise")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
