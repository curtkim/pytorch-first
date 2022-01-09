import os

from torch import nn, optim
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib import CIFAR10, Compose, ImageToTensor, NormalizeImage, ResidualBlock


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def resnet9(in_channels: int, num_classes: int, size: int = 16):
    sz, sz2, sz4, sz8 = size, size * 2, size * 4, size * 8
    return nn.Sequential(
        conv_block(in_channels, sz),
        conv_block(sz, sz2, pool=True),
        ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))),
        conv_block(sz2, sz4, pool=True),
        conv_block(sz4, sz8, pool=True),
        ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))),
        nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(sz8, num_classes)
        ),
    )


if __name__ == "__main__":
    # experiment setup
    logdir = "./logdir2"
    num_epochs = 10

    # data
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
    loaders = {
        "train": DataLoader(train_data, batch_size=32, num_workers=4),
        "valid": DataLoader(valid_data, batch_size=32, num_workers=4),
    }

    # model, criterion, optimizer, scheduler
    model = resnet9(in_channels=3, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

    # model training
    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        ddp=True,   # <-- here is the trick,
        amp=False,  # <-- here is another trick ;)
    )