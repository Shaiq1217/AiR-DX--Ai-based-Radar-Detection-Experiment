from src.data_loader import get_dataloader, get_resnet_data_loader
from src.model import SimpleCNN
from src.training import train_model
from src.inference import run_inferece, get_test_loader, run_gradcam
from src.generate_spectrograms import generate
from src.resnet import ResnetModel
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from glob import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

def training_loop():
    # Generate simulated spectrograms 
    generate(samples=400)

    #get training data with augmentations from "data" folder
    train_loader, val_loader, class_names = get_dataloader("data")

    # Initialize and train the model
    model_cnn_train = SimpleCNN(num_classes=len(class_names))
    train_model(model_cnn_train, train_loader, val_loader, num_epochs=15)

    # Benchmark on Resnet
    model_resnet_train = ResnetModel(num_classes=len(class_names))
    train_model(model_resnet_train, train_loader, val_loader, num_epochs=15, metrics_file="resnet_metrics.csv", weights_path="resnet_weights.pth")

def inference(num_samples = 50, out_dir="test", noise_mult = [0.0, 1.0, 1.2, 1.4, 1.5]):
     # create test set spectrograms
    generate(path = out_dir, samples=num_samples)

    # Load cnn model architecture
    print("Running inference on CNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cnn_test = SimpleCNN(num_classes=3)  # Adjust num_classes if needed
    model_cnn_test.load_state_dict(torch.load("out/cnn_best_weights.pth", map_location=device))
    model_cnn_test.to(device)
    model_cnn_test.eval()
    # Run inference and print metrics
    for multiplier in noise_mult:
        print(f"\n--- Running inference with noise multiplier: {multiplier} ---")
        run_inferece(model_cnn_test, device=device, noise_multiplier=multiplier,
                    confusion_matrix_path=f"out/cnn_confusion_matrix_noise_{multiplier}.png",
                    metrics_path="out/cnn_metrics_noise.csv")


    # Load resnet model architecture
    print("Running inference on Resnet model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_resnet_test = ResnetModel(num_classes=3)  # Adjust num_classes if needed
    model_resnet_test.load_state_dict(torch.load("out/resnet_weights.pth", map_location=device))
    model_resnet_test.to(device)
    model_resnet_test.eval()
    # Run inference and print metrics
    for multiplier in noise_mult:
        print(f"\n--- Running inference with noise multiplier: {multiplier} ---")
        run_inferece(model_resnet_test, device=device, noise_multiplier=multiplier,
                    confusion_matrix_path=f"out/resnet_confusion_matrix_noise_{multiplier}.png", metrics_path="out/resnet_metrics_noise.csv", num_channels=3)

def grad_cam(out_dir = './test_gradcam/',
    num_samples = 5, gradcam_out = 'out/gradcam_cnn', noise_multiplier = 1.0):
    #Test with gradcam
    generate(path=out_dir, samples=num_samples)
    print(f"[✓] Generated test set")
    test_loader, classes = get_test_loader(path= out_dir, num_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cnn_test = SimpleCNN(num_classes=3)  # Adjust num_classes if needed
    model_cnn_test.load_state_dict(torch.load("out/cnn_best_weights.pth", map_location=device))
    model_cnn_test.to(device)
    print(f"[✓] Loaded model")
    # run through gradcam function
    run_gradcam(model_cnn_test, 'cpu', 1, test_loader, classes, noise_multiplier=noise_multiplier, gradcam_output_dir=f"{gradcam_out}_noise_{noise_multiplier}")

def main():
    inference()


if __name__ == "__main__":
    main()
