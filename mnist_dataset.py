from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break
