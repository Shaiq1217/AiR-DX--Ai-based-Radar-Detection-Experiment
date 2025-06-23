from src.data_loader import get_dataloader
from src.model import SimpleCNN
from src.training import train_model
from src.inference import run_inferece
from src.generate_spectrograms import generate

import torch

def main():
    # generate(samples=400)
    # train_loader, val_loader, class_names = get_dataloader("data")
    # model = SimpleCNN(num_classes=len(class_names))
    # train_model(model, train_loader, val_loader, num_epochs=15)

    # run inference
    generate(path = "test", samples=50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Load model architecture
    model = SimpleCNN(num_classes=3)  # Adjust num_classes if needed
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Run inference and print metrics
    run_inferece(model, device=device)

if __name__ == "__main__":
    main()


