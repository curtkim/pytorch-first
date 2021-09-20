import torch
from torch.utils.tensorboard import SummaryWriter

with SummaryWriter() as w:
    for i in range(5):
        w.add_hparams(
            {'lr': 0.1*i, 'bsize': i},
            {'hparam/accuracy': 10*i, 'hparam/loss': 10*i}
        )