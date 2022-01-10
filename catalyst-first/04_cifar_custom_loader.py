import os

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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

class CustomSupervisedRunner(dl.SupervisedRunner):
    # here is the trick:
    def get_loaders(self, stage: str):
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
        if self.engine.is_ddp:
            train_sampler = DistributedSampler(
                train_data,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=True,
            )
            valid_sampler = DistributedSampler(
                valid_data,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=False,
            )
        else:
            train_sampler = valid_sampler = None

        train_loader = DataLoader(
            train_data, batch_size=32*32, sampler=train_sampler, num_workers=4
        )
        valid_loader = DataLoader(
            valid_data, batch_size=32*32, sampler=valid_sampler, num_workers=4
        )
        return {"train": train_loader, "valid": valid_loader}


if __name__ == "__main__":
    # experiment setup
    logdir = "./log_04_cifar_custom_loader"
    num_epochs = 2

    # model, criterion, optimizer, scheduler
    model = resnet9(in_channels=3, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

    # model training
    runner = CustomSupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=None,  # <-- here is the trick
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        ddp=True,   # <-- now it works like a charm
        amp=False,  # <-- you can still use this trick here ;)
    )
