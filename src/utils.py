import torch
import numpy as np

def get_conv_output(model, input_shape):
    """Compute the flattened output size of a CNN model after conv layers."""
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)  # e.g. (1, 224, 224)
        x = model._forward_conv(x)
        return int(np.prod(x.size()))
