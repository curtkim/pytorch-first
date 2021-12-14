from typing import List, Any
import logging
import warnings
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.nn import functional as F
from pytorch_lightning import loggers as pl_loggers
from mnist_datamodule import MNISTDataModule
from pytorch_lightning.utilities import rank_zero_only


class MyModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train/my_loss1', loss, prog_bar=True)
        self.log('train/my_loss2', loss, prog_bar=True)

        return {'loss': loss}
        #tensorboard_logs = {'train_loss': loss}
        #return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val/loss', loss)

        return {"loss": loss}
        #return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)



def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

if __name__ == '__main__':
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     handlers=[
    #         logging.FileHandler("debug.log"),
    #         logging.StreamHandler()
    #     ]
    # )

    # # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    # # configure logging on module level, redirect to file
    # logger = logging.getLogger("pytorch_lightning.core")
    # logger.addHandler(logging.FileHandler("core.log"))

    #csv_logger = pl_loggers.CSVLogger("logs_csv")
    tb_logger = pl_loggers.TensorBoardLogger("logs_tb", name=None)  # default logger와 동일하게 출력한다.

    log = get_logger(__name__)
    log.info("Disabling python warnings! <config.ignore_warnings=True>")
    warnings.filterwarnings("ignore")

    log.info("start")
    model = MyModel()
    datamodule = MNISTDataModule()
    trainer = Trainer(gpus=1, num_nodes=1, max_epochs=1, logger=tb_logger)
    trainer.fit(model=model, datamodule=datamodule)
    log.info("end")
