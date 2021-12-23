import logging
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torch
from typing import Optional, List, Any


class DummyDataset(Dataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        idx = self.start + index
        return (
            torch.tensor([idx, 2 * idx, 3 * idx], dtype=torch.float32),
            torch.tensor(idx, dtype=torch.float32)
        )


class DummyDataModule(LightningDataModule):
    def __init__(self, train_size, val_size, batch_size, num_workers):
        self.data_train = DummyDataset(0, train_size)
        self.data_val = DummyDataset(train_size, train_size+val_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = False

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


class DummyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return {'loss': y}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        return {'loss': y}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':
    logging.info('start')

    model = DummyModel()
    datamodule = DummyDataModule(100, 20, batch_size=2, num_workers=0)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model=model, datamodule=datamodule)

    logging.end('start')
