import os
import sys
import torch
from torch import tensor
from torchmetrics import Metric


MAX_PORT = 8100
START_PORT = 8088
CURRENT_PORT = START_PORT


def setup_ddp(rank, world_size):
    """Setup ddp environment."""
    global CURRENT_PORT

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(CURRENT_PORT)

    CURRENT_PORT += 1
    if CURRENT_PORT > MAX_PORT:
        CURRENT_PORT = START_PORT

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


class DummyMetric(Metric):
    name = "Dummy"

    def __init__(self):
        super().__init__()
        self.add_state("x", tensor(0.0), dist_reduce_fx=None)

    def update(self):
        pass

    def compute(self):
        pass

# from https://github.com/PyTorchLightning/metrics/blob/43a226101e05ec8e5a5fe4ad64b90c44af1a3b34/tests/bases/test_ddp.py
def _test_ddp_sum(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = DummyMetric()
    dummy._reductions = {"foo": torch.sum}
    dummy.foo = tensor(1)
    dummy._sync_dist()

    assert dummy.foo == worldsize
    print(os.getpid(), dummy.foo)


def _test_ddp_cat(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = DummyMetric()
    dummy._reductions = {"foo": torch.cat}
    dummy.foo = [tensor([1])]
    dummy._sync_dist()

    assert torch.all(torch.eq(dummy.foo, tensor([1, 1])))


if __name__ == '__main__':
    torch.multiprocessing.spawn(_test_ddp_sum, args=(2,), nprocs=2)
    torch.multiprocessing.spawn(_test_ddp_cat, args=(2,), nprocs=2)
