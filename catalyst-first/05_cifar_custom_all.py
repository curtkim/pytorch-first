import os

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from catalyst import dl, utils
from catalyst.contrib import CIFAR10, Compose, ImageToTensor, NormalizeImage, ResidualBlock

import argparse


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

class CustomRunner(dl.IRunner):
    def __init__(self, logdir: str, master_addr: str, master_port: int, world_size: int, workers_dist_rank: int, num_node_workers: int):
        super().__init__()
        self._logdir = logdir
        self.master_addr = master_addr
        self.master_port = master_port
        self.world_size = world_size
        self.workers_dist_rank = workers_dist_rank
        self.num_node_workers = num_node_workers

    def get_engine(self):
        #return dl.DistributedDataParallelAMPEngine()
        return dl.DistributedDataParallelEngine(
                    address=self.master_addr,
                    port=self.master_port,
                    ddp_kwargs={"find_unused_parameters": False},
                    process_group_kwargs={"backend": "nccl"},

                    world_size=self.world_size,
                    workers_dist_rank=self.workers_dist_rank,
                    num_node_workers=self.num_node_workers,
                )
        #return dl.DataParallelEngine()

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }

    @property
    def stages(self):
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 2

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

    def get_model(self, stage: str):
        model = (
            self.model
            if self.model is not None
            else resnet9(in_channels=3, num_classes=10)
        )
        return model

    def get_criterion(self, stage: str):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, stage: str, model):
        return optim.Adam(model.parameters(), lr=1e-3)

    def get_scheduler(self, stage: str, optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

    def get_callbacks(self, stage: str):
        return {
            "criterion": dl.CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            "accuracy": dl.AccuracyCallback(
                input_key="logits", target_key="targets", topk_args=(1, 3, 5)
            ),
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="loss",
                minimize=False,
                save_n_best=1,
            ),
            # "tqdm": dl.TqdmCallback(),
        }

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)
        self.batch = {
            "features": x,
            "targets": y,
            "logits": logits
        }


if __name__ == "__main__":
    # multi node test를 node 하나에서 실행해보기 (동작하지 않음, node에서는 spawn할때 rank가 0부터 시작하는 것이 강제된다)
    # python 05_cifar_custom_all.py --world_size=2 --workers_dist_rank=0 --num_node_workers=1
    # python 05_cifar_custom_all.py --world_size=2 --workers_dist_rank=1 --num_node_workers=1

    # multi node, each node has 2 gpus
    # python 05_cifar_custom_all.py --master_addr=a.b.c --master_port=31088 --world_size=4 --workers_dist_rank=0 --num_node_workers=2
    # python 05_cifar_custom_all.py --master_addr=a.b.c --master_port=31088 --world_size=4 --workers_dist_rank=2 --num_node_workers=2

    # single node 2 gpu
    # python 05_cifar_custom_all.py --master_addr=0.0.0.0 --master_port=20000 --world_size=2 --workers_dist_rank=0 --num_node_workers=2

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--master_addr', type=str)
    parser.add_argument('--master_port', type=int)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--workers_dist_rank', type=int)
    parser.add_argument('--num_node_workers', type=int)
    args = parser.parse_args()

    # experiment setup
    logdir = "./log_05_cifar_custom_all"

    runner = CustomRunner(logdir, args.master_addr, args.master_port, args.world_size, args.workers_dist_rank, args.num_node_workers)
    runner.run()


