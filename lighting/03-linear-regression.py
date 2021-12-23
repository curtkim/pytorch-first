import sys
import logging
import warnings
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning
from pytorch_lightning.callbacks import ProgressBarBase
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging_tree


class MyDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]


class MyDataModule(LightningDataModule):
    def __init__(self, data_size, batch_size, num_workers, pin_memory):
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        # create dummy data for training
        x_values = [i for i in range(self.data_size)]
        x_train = np.array(x_values, dtype=np.float32)
        x_train = x_train.reshape(-1, 1)

        y_values = [2 * i + 1 for i in x_values]
        y_train = np.array(y_values, dtype=np.float32)
        y_train = y_train.reshape(-1, 1)

        self.data_train = MyDataset(x_train, y_train)

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )


class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


class MyModule(LightningModule):
    def __init__(self, learningRate):
        super().__init__()
        self.learningRate = learningRate
        self.save_hyperparameters()

        self.model = LinearRegression(1, 1)
        self.criterion = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs: List[Any]):
        # print(self.current_epoch, outputs)
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.learningRate)


class ConsoleProgressBar(ProgressBarBase):
    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
        percent = (self.train_batch_idx / self.total_train_batches) * 100
        #sys.stdout.flush()
        #sys.stdout.write(f'{percent:.01f} percent complete \r')
        #print(trainer.current_epoch, batch_idx, outputs)
        #print(f'{percent:.01f} percent complete {batch_idx}')

    def on_epoch_end(self, trainer, pl_module):
        pass


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    pytorch_lightning_logger = logging.getLogger('pytorch_lightning')
    pytorch_lightning_logger.propagate = True
    pytorch_lightning_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)s %(name)s %(funcName)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    #logging_tree.printout()

    logging.info('start')
    module = MyModule(0.0001)
    datamodule = MyDataModule(20, batch_size=10, num_workers=0, pin_memory=False)

    # callbacks = [
    #     pytorch_lightning.callbacks.model_summary.ModelSummary(),
    #     pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint(),
    #     #pytorch_lightning.callbacks.progress.RichProgressBar(refresh_rate_per_second=0.01),
    #     ConsoleProgressBar(),
    # ]
    trainer = Trainer(
        max_epochs=100,
        #enable_progress_bar=True,
        # log_every_n_steps=10,
        # flush_logs_every_n_steps=10,
        #progress_bar_refresh_rate=0,
        #callbacks=callbacks,
    )
    # print(trainer.callbacks)
    # [
    #     pytorch_lightning.callbacks.model_summary.ModelSummary
    #     pytorch_lightning.callbacks.gradient_accumulation_scheduler.GradientAccumulationScheduler
    #     pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
    # ]

    trainer.fit(model=module, datamodule=datamodule)
    logging.info('end')

    '''
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)
    x_train = torch.from_numpy(x_train)

    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)
    
    with torch.no_grad():
        predicted = module.model(x_train).data.numpy()

    plt.clf()
    plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
    plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()
    '''