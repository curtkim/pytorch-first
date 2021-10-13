import torch
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self, use_memory_efficient):
        super(MyModule, self).__init__()
        self.use_memory_efficient = use_memory_efficient

    @torch.jit.unused
    def memory_efficient(self, x):
        import pdb
        pdb.set_trace()
        return x + 10

    def forward(self, x):
        # Use not-yet-scriptable memory efficient mode
        if self.use_memory_efficient:
            return self.memory_efficient(x)
        else:
            return x + 10


m = torch.jit.script(MyModule(use_memory_efficient=False))
m(torch.rand(100))
m.save("m.pt")

m2 = torch.jit.script(MyModule(use_memory_efficient=True))
m2.save("m2.pt")
try:
    m2(torch.rand(100))
except torch.jit.Error as err:
    print("torch.jit.Error raised")
