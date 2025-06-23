from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloader(path, batch_size = 8, split_ratio=0.8):
  # Define transforms
  transform = transforms.Compose([
      transforms.Grayscale(),
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])

  # Load the full dataset
  full_dataset = datasets.ImageFolder(path, transform=transform)

  # Split into 80% train, 20% val
  train_size = int(split_ratio * len(full_dataset))
  val_size = len(full_dataset) - train_size
  train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

  # Dataloaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  # Class names (needed for labels)
  class_names = full_dataset.classes
  return train_loader, val_loader, class_names