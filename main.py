from src.data_loader import get_dataloader, get_resnet_data_loader
from src.model import SimpleCNN
from src.training import train_model
from src.inference import run_inferece
from src.generate_spectrograms import generate
from src.resnet import ResnetModel
from src.utils import plot_metrics
import torch

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

def inference(num_samples = 50):
     # create test set spectrograms
    generate(path = "test", samples=num_samples)

    # Load cnn model architecture
    print("Running inference on CNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cnn_test = SimpleCNN(num_classes=3)  # Adjust num_classes if needed
    model_cnn_test.load_state_dict(torch.load("out/cnn_best_weights.pth", map_location=device))
    model_cnn_test.to(device)
    model_cnn_test.eval()
    # Run inference and print metrics
    for multiplier in [0.0, 0.25, 0.5, 1.0]:
        print(f"\n--- Running inference with noise multiplier: {multiplier} ---")
        run_inferece(model_cnn_test, device='cuda', noise_multiplier=multiplier,
                    confusion_matrix_path=f"out/confusion_matrix_noise_{multiplier}.png")


    # Load resnet model architecture
    print("Running inference on Resnet model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_resnet_test = ResnetModel(num_classes=3)  # Adjust num_classes if needed
    model_resnet_test.load_state_dict(torch.load("out/resnet_weights.pth", map_location=device))
    model_resnet_test.to(device)
    model_resnet_test.eval()
    # Run inference and print metrics
    for multiplier in [0.0, 0.25, 0.5, 1.0]:
        print(f"\n--- Running inference with noise multiplier: {multiplier} ---")
        run_inferece(model_resnet_test, device='cuda', noise_multiplier=multiplier,
                    confusion_matrix_path=f"out/confusion_matrix_noise_{multiplier}.png")



def main():
    #Increase noise in test set
    inference()

if __name__ == "__main__":
    main()

