import torch
from pytorch_lightning import Trainer
from mymodel import LitModel


if __name__ == '__main__':
    model = LitModel()
    trainer = Trainer(gpus=1, num_nodes=1)
    trainer.fit(model)
    trainer.save_checkpoint("example2.ckpt")
