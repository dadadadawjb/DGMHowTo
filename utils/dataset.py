import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def load_mnist(path:str, batch_size:int):
    # batch_x: torch.Tensor((batch_size, 1, 28, 28)D), N(0.0, 1.0)
    # batch_y: torch.Tensor((1, batch_size)D), [0, 9]
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,)), 
    ])

    train_data = datasets.MNIST(root=path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader
