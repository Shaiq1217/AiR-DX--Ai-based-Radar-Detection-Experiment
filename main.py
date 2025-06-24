from src.data_loader import get_dataloader, get_resnet_data_loader
from src.model import SimpleCNN
from src.training import train_model
from src.inference import run_inferece
from src.generate_spectrograms import generate
from src.resnet import ResnetModel
import torch

def main():
    # Generate simulated spectrograms 
    generate(samples=400)

    #get training data with augmentations from "data" folder
    train_loader, val_loader, class_names = get_dataloader("data")

    # Initialize and train the model
    model = SimpleCNN(num_classes=len(class_names))
    train_model(model, train_loader, val_loader, num_epochs=15)

    # Benchmark on Resnet
    model = ResnetModel(num_classes=len(class_names))
    train_model(model, train_loader, val_loader, num_epochs=15, metrics_file="resnet_metrics.csv", weights_path="resnet_weights.pth")

    # create test set spectrograms
    generate(path = "test", samples=50)

    # Load cnn model architecture
    print("Running inference on CNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=3)  # Adjust num_classes if needed
    model.load_state_dict(torch.load("out/cnn_best_weights.pth", map_location=device))
    model.to(device)
    model.eval()
    # Run inference and print metrics
    run_inferece(model, device=device, num_channels=1)


    # Load cnn model architecture
    print("Running inference on Resnet model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=3)  # Adjust num_classes if needed
    model.load_state_dict(torch.load("out/resnet_weights.pth", map_location=device))
    model.to(device)
    model.eval()
    # Run inference and print metrics
    run_inferece(model, device=device, num_channels=1)

    #Increase noise in test set
    #generate(path = "test", samples=50, noise_level=0.1)


if __name__ == "__main__":
    main()


