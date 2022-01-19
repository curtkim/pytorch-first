import torch
from pytorch_lightning import Trainer
from mymodel import LitModel


if __name__ == '__main__':
    #model = LitModel.load_from_checkpoint("example.ckpt")
    model = LitModel()

    trainer = Trainer(gpus=1, num_nodes=1, resume_from_checkpoint="example.ckpt")
    trainer.fit(model)
    trainer.save_checkpoint("example_resume.ckpt")
