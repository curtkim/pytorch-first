import os

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from catalyst import dl
from catalyst.contrib import CIFAR10, Compose, ImageToTensor, NormalizeImage, ResidualBlock


transform = Compose([
    ImageToTensor(),
    NormalizeImage((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = CIFAR10(
    os.getcwd(), train=True, download=True, transform=transform
)
valid_data = CIFAR10(
    os.getcwd(), train=False, download=True, transform=transform
)


print('len(train_data)', len(train_data))   # 50000
print('len(valid_data)', len(valid_data))   # 10000
