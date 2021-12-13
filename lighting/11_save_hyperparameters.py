from pytorch_lightning import LightningModule, Trainer
from torch import nn
import torch


class LitModel(LightningModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        print("before save_hyperparameters", self.hparams)
        self.save_hyperparameters()
        print("after save_hyperparameters", self.hparams)
        self.l1 = nn.Linear(self.hparams.in_dim, self.hparams.out_dim)


if __name__ == '__main__':
    model = LitModel(in_dim=32, out_dim=10)

    # CKPT_PATH = "example.ckpt"
    # trainer = Trainer()
    # trainer.fit(model)
    # trainer.save_checkpoint(CKPT_PATH)
    #
    # checkpoint = torch.load(CKPT_PATH)
    # print(checkpoint.keys())