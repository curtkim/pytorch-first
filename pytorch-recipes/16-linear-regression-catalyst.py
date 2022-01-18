# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
import logging
import os

import numpy as np
import torch
from catalyst.core import Callback, CallbackOrder, IRunner
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from catalyst import dl

print('os.getpid()', os.getpid())


class MyDataset(Dataset):
    def __init__(self, size):
        self.size = size

        # create dummy data for training
        x_values = [i for i in range(self.size)]
        x_train = np.array(x_values, dtype=np.float32)
        self.x_train = x_train.reshape(-1, 1)

        y_values = [2 * i + 1 for i in x_values]
        y_train = np.array(y_values, dtype=np.float32)
        self.y_train = y_train.reshape(-1, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


dataset = MyDataset(10)
loaders = {
    "train": torch.utils.data.DataLoader(dataset, batch_size=2)
}

inputDim = 1        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 0.01
num_epochs = 1

model = linearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)


class DebugCallback(Callback):
    def __init__(self):
        super().__init__(order=CallbackOrder.external)

    def on_stage_start(self, runner: IRunner):
        ## logging config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"debug_{runner.engine.world_size}_{runner.engine.rank}.log", 'w')
            ]
        )

    def on_batch_start(self, runner: "IRunner") -> None:
        assert 2 == len(runner.batch.keys())
        assert 'inputs' in runner.batch
        assert 'targets' in runner.batch

        logging.info(runner.batch['inputs'].cpu())

    def on_batch_end(self, runner: "IRunner") -> None:
        assert 3 == len(runner.batch.keys())
        assert 'inputs' in runner.batch
        assert 'targets' in runner.batch
        assert 'outputs' in runner.batch

        assert 'loss' in runner.batch_metrics


runner = dl.SupervisedRunner(input_key="inputs", target_key="targets", output_key="outputs")
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir="logs",
    num_epochs=num_epochs,
    verbose=True,
    callbacks=[
        DebugCallback()
    ]
)

model_cpu = model.cpu()
with torch.no_grad():   # we don't need gradients in the testing phase
    predicted = model_cpu(torch.from_numpy(dataset.x_train)).data.numpy()
    print(predicted)


plt.clf()
plt.plot(dataset.x_train, dataset.y_train, 'go', label='True data', alpha=0.5)
plt.plot(dataset.x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
