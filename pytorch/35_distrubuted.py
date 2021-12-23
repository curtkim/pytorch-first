# optimizer.step() 전에 average_gradients를 호출하는게 핵심인것 같다.
# model.parameters()들을 평균낸다.

import os
import logging
import typing

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from math import ceil
from random import Random
from torch.autograd import Variable
from torchvision import datasets, transforms


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index: typing.List):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        #rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
        print('self.partitions', type(self.partitions), len(self.partitions))
        print('self.data', type(self.data), len(self.data))
        for partition in self.partitions:
            print('min', min(partition), 'max', max(partition))

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def partition_dataset(rank: int, world_size: int):
    """ Partitioning MNIST
    """

    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    batch_size = int(128 / world_size)
    partition_sizes = [1.0 / world_size for _ in range(world_size)]

    partition = DataPartitioner(dataset, partition_sizes).use(rank)
    train_set = torch.utils.data.DataLoader(partition, batch_size=batch_size, shuffle=True)
    return train_set, batch_size


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        ### 중요
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run(): #rank: int, size: int
    """ Distributed Synchronous SGD Example """

    # process별로 log파일을 가진다.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)s %(funcName)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"debug{dist.get_rank()}.log", "w")
        ]
    )

    torch.manual_seed(1234)
    train_set, batch_size = partition_dataset(dist.get_rank(), dist.get_world_size())
    model = Net()
    #    model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(batch_size))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            #data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data
            loss.backward()

            ### 중요
            average_gradients(model)
            optimizer.step()
        logging.info(f"epoch {epoch} Rank {dist.get_rank()}({os.getpid()}) loss = {epoch_loss / num_batches}")


def init_processes(rank, world_size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn() #rank, world_size


if __name__ == "__main__":

    world_size = 2
    processes = []

    for rank in range(world_size):
        p = mp.Process(target=init_processes, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
